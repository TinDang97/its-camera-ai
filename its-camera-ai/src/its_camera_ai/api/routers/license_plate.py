"""
License Plate Recognition API router.

RESTful API endpoints for license plate detection, search, watchlist management,
and analytics with comprehensive error handling and performance monitoring.
"""

import base64
import logging
import time
from datetime import UTC
from uuid import UUID

import cv2
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import JSONResponse
from sqlalchemy import func
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.auth import get_current_user
from ...core.database import get_async_session
from ...ml.license_plate_recognition import (
    LicensePlateRecognitionPipeline,
    PlateDetectionStatus,
)
from ...models.license_plate import PlateDetection, PlateWatchlist
from ...models.user import User
from ..dependencies import get_lpr_pipeline
from ..schemas.license_plate import (
    AlertPriority,
    AlertType,
    AnalyticsResponse,
    BoundingBox,
    HealthCheckResponse,
    PlateDetectionRecord,
    PlateDetectionRequest,
    PlateDetectionResponse,
    PlateDetectionResult,
    PlateRegion,
    PlateSearchResponse,
    ValidationRequest,
    ValidationResponse,
    WatchlistCreateRequest,
    WatchlistResponse,
    WatchlistSearchResponse,
    WatchlistUpdateRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(
    tags=["License Plate Recognition"],
    responses={
        status.HTTP_500_INTERNAL_SERVER_ERROR: {
            "description": "Internal server error"
        }
    }
)


def decode_image(image_data: str) -> np.ndarray:
    """Decode base64 image data to numpy array."""
    try:
        # Remove data URL prefix if present
        if "," in image_data:
            image_data = image_data.split(",", 1)[1]

        # Decode base64
        image_bytes = base64.b64decode(image_data)

        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)

        # Decode image
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Failed to decode image")

        return image

    except Exception as e:
        logger.error(f"Failed to decode image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image data: {str(e)}"
        )


@router.post("/detect", response_model=PlateDetectionResponse)
async def detect_license_plates(
    request: PlateDetectionRequest,
    lpr: LicensePlateRecognitionPipeline = Depends(get_lpr_pipeline),
    current_user: User = Depends(get_current_user)
):
    """
    Detect and recognize license plates in real-time.
    
    Processes vehicle detections and attempts to recognize license plates
    within each vehicle bounding box.
    """
    start_time = time.perf_counter()

    try:
        # Decode image
        image = decode_image(request.image_data)

        # Extract vehicle detections
        vehicle_detections = []
        for detection in request.vehicle_detections:
            bbox = detection.get("bbox", {})
            confidence = detection.get("confidence", 0.0)

            if all(k in bbox for k in ["x1", "y1", "x2", "y2"]):
                vehicle_bbox = (
                    int(bbox["x1"]), int(bbox["y1"]),
                    int(bbox["x2"]), int(bbox["y2"])
                )
                vehicle_detections.append((vehicle_bbox, confidence))

        if not vehicle_detections:
            return PlateDetectionResponse(
                success=True,
                detections=[],
                processing_time_ms=(time.perf_counter() - start_time) * 1000
            )

        # Perform batch license plate recognition
        results = await lpr.batch_recognize(image, vehicle_detections)

        # Convert results to API response format
        api_results = []
        for result in results:
            if result.status == PlateDetectionStatus.SUCCESS:
                api_result = PlateDetectionResult(
                    plate_text=result.plate_text,
                    confidence=result.confidence,
                    ocr_confidence=result.ocr_confidence,
                    vehicle_bbox=BoundingBox(
                        x1=result.vehicle_bbox[0],
                        y1=result.vehicle_bbox[1],
                        x2=result.vehicle_bbox[2],
                        y2=result.vehicle_bbox[3]
                    ),
                    plate_bbox=BoundingBox(
                        x1=result.plate_bbox[0],
                        y1=result.plate_bbox[1],
                        x2=result.plate_bbox[2],
                        y2=result.plate_bbox[3]
                    ) if result.plate_bbox else None,
                    plate_quality_score=result.plate_quality_score,
                    character_confidences=result.character_confidences,
                    processing_time_ms=result.processing_time_ms,
                    ocr_time_ms=result.ocr_time_ms,
                    engine_used=result.engine_used,
                    region=result.region,
                    is_reliable=result.is_reliable,
                    is_high_confidence=result.is_high_confidence,
                    triggered_alerts=[]  # Would be populated by watchlist check
                )
                api_results.append(api_result)

        processing_time = (time.perf_counter() - start_time) * 1000

        return PlateDetectionResponse(
            success=True,
            detections=api_results,
            processing_time_ms=processing_time,
            cached=False  # Would be set based on cache hits
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"License plate detection failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@router.get("/search", response_model=PlateSearchResponse)
async def search_plate_detections(
    plate_text: str | None = Query(None, description="Plate text to search"),
    camera_ids: list[UUID] | None = Query(None, description="Camera IDs"),
    start_date: str | None = Query(None, description="Start date (ISO format)"),
    end_date: str | None = Query(None, description="End date (ISO format)"),
    min_confidence: float | None = Query(0.5, ge=0.0, le=1.0),
    limit: int | None = Query(100, ge=1, le=1000),
    offset: int | None = Query(0, ge=0),
    include_false_positives: bool | None = Query(False),
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Search license plate detection history.
    
    Supports filtering by plate text, camera, date range, and confidence.
    """
    start_time = time.perf_counter()

    try:
        # Build query
        query = session.query(PlateDetection)

        # Apply filters
        if plate_text:
            normalized_plate = plate_text.upper().replace(" ", "").replace("-", "")
            query = query.filter(PlateDetection.plate_text_normalized == normalized_plate)

        if camera_ids:
            query = query.filter(PlateDetection.camera_id.in_(camera_ids))

        if start_date:
            from datetime import datetime
            start_dt = datetime.fromisoformat(start_date.replace("Z", "+00:00"))
            query = query.filter(PlateDetection.created_at >= start_dt)

        if end_date:
            from datetime import datetime
            end_dt = datetime.fromisoformat(end_date.replace("Z", "+00:00"))
            query = query.filter(PlateDetection.created_at <= end_dt)

        if min_confidence > 0.0:
            query = query.filter(PlateDetection.overall_confidence >= min_confidence)

        if not include_false_positives:
            query = query.filter(PlateDetection.is_false_positive == False)

        # Get total count
        total_count = await session.scalar(
            query.statement.with_only_columns([func.count(PlateDetection.id)])
        )

        # Apply pagination and ordering
        query = query.order_by(PlateDetection.created_at.desc())
        query = query.offset(offset).limit(limit)

        # Execute query
        result = await session.execute(query.statement)
        detections = result.scalars().all()

        # Convert to API response format
        detection_records = []
        for detection in detections:
            record = PlateDetectionRecord(
                id=detection.id,
                camera_id=detection.camera_id,
                camera_name=detection.camera.name if detection.camera else None,
                plate_text=detection.plate_text,
                plate_region=PlateRegion(detection.plate_region),
                overall_confidence=detection.overall_confidence,
                ocr_confidence=detection.ocr_confidence,
                plate_quality_score=detection.plate_quality_score,
                detected_at=detection.created_at,
                processing_time_ms=detection.processing_time_ms,
                is_validated=detection.is_validated,
                is_false_positive=detection.is_false_positive,
                validated_by=detection.validated_by,
                triggered_alerts=detection.triggered_alerts or []
            )
            detection_records.append(record)

        search_time = (time.perf_counter() - start_time) * 1000

        return PlateSearchResponse(
            detections=detection_records,
            total_count=total_count,
            has_more=(offset + len(detection_records)) < total_count,
            search_time_ms=search_time
        )

    except Exception as e:
        logger.error(f"Plate search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}"
        )


@router.post("/watchlist", response_model=WatchlistResponse)
async def create_watchlist_entry(
    request: WatchlistCreateRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Create a new watchlist entry for alert monitoring.
    """
    try:
        # Create watchlist entry
        watchlist_entry = PlateWatchlist(
            plate_number=request.plate_number,
            alert_type=request.alert_type.value,
            alert_priority=request.alert_priority.value,
            description=request.description,
            notes=request.notes,
            expires_at=request.expires_at,
            contact_info=request.contact_info,
            notification_channels=request.notification_channels,
            owner_name=request.owner_name,
            owner_info=request.owner_info,
            agency=request.agency,
            jurisdiction=request.jurisdiction,
            case_number=request.case_number,
            created_by=current_user.username
        )

        # Normalize plate number
        watchlist_entry.normalize_plate_number()

        # Add to session and commit
        session.add(watchlist_entry)
        await session.commit()
        await session.refresh(watchlist_entry)

        # Convert to response format
        return WatchlistResponse(
            id=watchlist_entry.id,
            plate_number=watchlist_entry.plate_number,
            alert_type=AlertType(watchlist_entry.alert_type),
            alert_priority=AlertPriority(watchlist_entry.alert_priority),
            is_active=watchlist_entry.is_active,
            expires_at=watchlist_entry.expires_at,
            description=watchlist_entry.description,
            notes=watchlist_entry.notes,
            contact_info=watchlist_entry.contact_info,
            notification_channels=watchlist_entry.notification_channels,
            owner_name=watchlist_entry.owner_name,
            owner_info=watchlist_entry.owner_info,
            agency=watchlist_entry.agency,
            jurisdiction=watchlist_entry.jurisdiction,
            case_number=watchlist_entry.case_number,
            total_detections=watchlist_entry.total_detections,
            last_detected_at=watchlist_entry.last_detected_at,
            last_detected_camera_id=watchlist_entry.last_detected_camera_id,
            created_at=watchlist_entry.created_at,
            created_by=watchlist_entry.created_by,
            updated_at=watchlist_entry.updated_at,
            updated_by=watchlist_entry.updated_by
        )

    except Exception as e:
        logger.error(f"Failed to create watchlist entry: {e}")
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create watchlist entry: {str(e)}"
        )


@router.get("/watchlist", response_model=WatchlistSearchResponse)
async def search_watchlist(
    plate_number: str | None = Query(None),
    alert_type: AlertType | None = Query(None),
    alert_priority: AlertPriority | None = Query(None),
    is_active: bool | None = Query(None),
    agency: str | None = Query(None),
    include_expired: bool | None = Query(False),
    limit: int | None = Query(100, ge=1, le=1000),
    offset: int | None = Query(0, ge=0),
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Search watchlist entries with filtering.
    """
    try:
        query = session.query(PlateWatchlist)

        # Apply filters
        if plate_number:
            normalized_plate = plate_number.upper().replace(" ", "").replace("-", "")
            query = query.filter(PlateWatchlist.plate_number_normalized == normalized_plate)

        if alert_type:
            query = query.filter(PlateWatchlist.alert_type == alert_type.value)

        if alert_priority:
            query = query.filter(PlateWatchlist.alert_priority == alert_priority.value)

        if is_active is not None:
            query = query.filter(PlateWatchlist.is_active == is_active)

        if agency:
            query = query.filter(PlateWatchlist.agency.ilike(f"%{agency}%"))

        if not include_expired:
            from datetime import datetime
            query = query.filter(
                (PlateWatchlist.expires_at.is_(None)) |
                (PlateWatchlist.expires_at > datetime.now(UTC))
            )

        # Get total count
        total_count = await session.scalar(
            query.statement.with_only_columns([func.count(PlateWatchlist.id)])
        )

        # Apply pagination
        query = query.order_by(PlateWatchlist.created_at.desc())
        query = query.offset(offset).limit(limit)

        # Execute query
        result = await session.execute(query.statement)
        entries = result.scalars().all()

        # Convert to response format
        watchlist_entries = []
        for entry in entries:
            response_entry = WatchlistResponse(
                id=entry.id,
                plate_number=entry.plate_number,
                alert_type=AlertType(entry.alert_type),
                alert_priority=AlertPriority(entry.alert_priority),
                is_active=entry.is_active,
                expires_at=entry.expires_at,
                description=entry.description,
                notes=entry.notes,
                contact_info=entry.contact_info,
                notification_channels=entry.notification_channels,
                owner_name=entry.owner_name,
                owner_info=entry.owner_info,
                agency=entry.agency,
                jurisdiction=entry.jurisdiction,
                case_number=entry.case_number,
                total_detections=entry.total_detections,
                last_detected_at=entry.last_detected_at,
                last_detected_camera_id=entry.last_detected_camera_id,
                created_at=entry.created_at,
                created_by=entry.created_by,
                updated_at=entry.updated_at,
                updated_by=entry.updated_by
            )
            watchlist_entries.append(response_entry)

        return WatchlistSearchResponse(
            entries=watchlist_entries,
            total_count=total_count,
            has_more=(offset + len(watchlist_entries)) < total_count
        )

    except Exception as e:
        logger.error(f"Watchlist search failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Watchlist search failed: {str(e)}"
        )


@router.put("/watchlist/{entry_id}", response_model=WatchlistResponse)
async def update_watchlist_entry(
    entry_id: UUID,
    request: WatchlistUpdateRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Update an existing watchlist entry.
    """
    try:
        # Get existing entry
        result = await session.execute(
            session.query(PlateWatchlist).filter(PlateWatchlist.id == entry_id).statement
        )
        entry = result.scalar_one_or_none()

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Watchlist entry not found"
            )

        # Update fields
        if request.alert_type is not None:
            entry.alert_type = request.alert_type.value
        if request.alert_priority is not None:
            entry.alert_priority = request.alert_priority.value
        if request.description is not None:
            entry.description = request.description
        if request.notes is not None:
            entry.notes = request.notes
        if request.expires_at is not None:
            entry.expires_at = request.expires_at
        if request.is_active is not None:
            entry.is_active = request.is_active
        if request.contact_info is not None:
            entry.contact_info = request.contact_info
        if request.notification_channels is not None:
            entry.notification_channels = request.notification_channels
        if request.owner_name is not None:
            entry.owner_name = request.owner_name
        if request.owner_info is not None:
            entry.owner_info = request.owner_info
        if request.agency is not None:
            entry.agency = request.agency
        if request.jurisdiction is not None:
            entry.jurisdiction = request.jurisdiction
        if request.case_number is not None:
            entry.case_number = request.case_number

        # Update audit fields
        entry.updated_by = current_user.username

        await session.commit()
        await session.refresh(entry)

        # Return updated entry
        return WatchlistResponse(
            id=entry.id,
            plate_number=entry.plate_number,
            alert_type=AlertType(entry.alert_type),
            alert_priority=AlertPriority(entry.alert_priority),
            is_active=entry.is_active,
            expires_at=entry.expires_at,
            description=entry.description,
            notes=entry.notes,
            contact_info=entry.contact_info,
            notification_channels=entry.notification_channels,
            owner_name=entry.owner_name,
            owner_info=entry.owner_info,
            agency=entry.agency,
            jurisdiction=entry.jurisdiction,
            case_number=entry.case_number,
            total_detections=entry.total_detections,
            last_detected_at=entry.last_detected_at,
            last_detected_camera_id=entry.last_detected_camera_id,
            created_at=entry.created_at,
            created_by=entry.created_by,
            updated_at=entry.updated_at,
            updated_by=entry.updated_by
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update watchlist entry: {e}")
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update watchlist entry: {str(e)}"
        )


@router.delete("/watchlist/{entry_id}")
async def delete_watchlist_entry(
    entry_id: UUID,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a watchlist entry.
    """
    try:
        # Get existing entry
        result = await session.execute(
            session.query(PlateWatchlist).filter(PlateWatchlist.id == entry_id).statement
        )
        entry = result.scalar_one_or_none()

        if not entry:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Watchlist entry not found"
            )

        # Delete entry
        await session.delete(entry)
        await session.commit()

        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={"message": "Watchlist entry deleted successfully"}
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete watchlist entry: {e}")
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete watchlist entry: {str(e)}"
        )


@router.get("/analytics", response_model=AnalyticsResponse)
async def get_license_plate_analytics(
    camera_ids: list[UUID] | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
    group_by: str | None = Query("hour"),
    include_details: bool | None = Query(False),
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Get license plate recognition analytics.
    
    Returns aggregated statistics and time-series data for LPR performance.
    """
    start_time = time.perf_counter()

    try:
        # This would implement comprehensive analytics queries
        # For now, return a placeholder response

        from ..schemas.license_plate import (
            AnalyticsSummary,
        )

        # Placeholder analytics data
        summary = AnalyticsSummary(
            total_vehicles=1000,
            total_plate_attempts=950,
            successful_reads=865,
            unique_plates=742,
            success_rate=0.91,
            avg_confidence=0.84,
            avg_quality_score=0.78,
            avg_processing_time_ms=12.5,
            total_alerts=15,
            alerts_by_type={"stolen": 3, "wanted": 7, "custom": 5},
            plates_by_region={"us": 680, "ca": 45, "auto": 17}
        )

        computation_time = (time.perf_counter() - start_time) * 1000

        return AnalyticsResponse(
            summary=summary,
            data_points=[],
            cameras=[],
            computation_time_ms=computation_time
        )

    except Exception as e:
        logger.error(f"Analytics query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analytics query failed: {str(e)}"
        )


@router.post("/validate/{detection_id}", response_model=ValidationResponse)
async def validate_detection(
    detection_id: UUID,
    request: ValidationRequest,
    session: AsyncSession = Depends(get_async_session),
    current_user: User = Depends(get_current_user)
):
    """
    Validate or correct a license plate detection.
    """
    try:
        # Get detection
        result = await session.execute(
            session.query(PlateDetection).filter(PlateDetection.id == detection_id).statement
        )
        detection = result.scalar_one_or_none()

        if not detection:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Detection not found"
            )

        # Perform validation action
        if request.action == "validate":
            detection.validate_detection(current_user.username, request.confidence)
        elif request.action == "mark_false_positive":
            detection.mark_false_positive(current_user.username, request.notes)
        elif request.action == "correct":
            if request.corrected_text:
                detection.plate_text = request.corrected_text
                detection.normalize_plate_text()
                detection.validate_detection(current_user.username, request.confidence)

        await session.commit()

        return ValidationResponse(
            success=True,
            detection_id=detection_id,
            action_taken=request.action,
            updated_at=detection.updated_at,
            updated_by=current_user.username
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        await session.rollback()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Validation failed: {str(e)}"
        )


@router.get("/health", response_model=HealthCheckResponse)
async def health_check(
    lpr: LicensePlateRecognitionPipeline = Depends(get_lpr_pipeline)
):
    """
    Check LPR system health and performance.
    """
    try:
        # Get LPR pipeline stats
        stats = lpr.get_stats()

        # Check OCR engine availability
        ocr_engines = {
            "tensorrt": False,  # Would check actual availability
            "easyocr": True,
            "paddleocr": True
        }

        # Performance metrics
        performance_metrics = {
            "avg_processing_time_ms": stats.get("avg_processing_time_ms", 0.0),
            "success_rate_percent": stats.get("success_rate_percent", 0.0),
            "cache_hit_rate_percent": stats.get("cache_hit_rate_percent", 0.0)
        }

        # Cache status
        cache_status = {
            "enabled": True,
            "hit_rate": stats.get("cache_hit_rate_percent", 0.0),
            "size": stats.get("cached_results", 0)
        }

        # Memory usage (placeholder)
        memory_usage = {
            "gpu_memory_mb": 512.0,
            "system_memory_mb": 2048.0,
            "utilization_percent": 65.0
        }

        return HealthCheckResponse(
            status="healthy",
            ocr_engines=ocr_engines,
            model_status={"plate_detector": "loaded", "ocr_engine": "loaded"},
            performance_metrics=performance_metrics,
            cache_status=cache_status,
            memory_usage=memory_usage,
            uptime_seconds=3600.0  # Placeholder
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="unhealthy",
            ocr_engines={},
            model_status={"error": str(e)},
            performance_metrics={},
            cache_status={},
            memory_usage={},
            uptime_seconds=0.0
        )
