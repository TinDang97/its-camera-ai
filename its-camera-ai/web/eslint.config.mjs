import { dirname } from "path";
import { fileURLToPath } from "url";
import { FlatCompat } from "@eslint/eslintrc";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

const compat = new FlatCompat({
  baseDirectory: __dirname,
});

const eslintConfig = [
  // Base Next.js configuration
  ...compat.extends("next/core-web-vitals", "next/typescript"),

  // Enhanced TypeScript and React 19 configuration
  ...compat.extends(
    "plugin:@typescript-eslint/strict-type-checked",
    "plugin:@typescript-eslint/stylistic-type-checked",
    "plugin:react-hooks/recommended",
    "plugin:jsx-a11y/recommended",
    "plugin:import/recommended",
    "plugin:import/typescript"
  ),

  {
    languageOptions: {
      parserOptions: {
        project: "./tsconfig.json",
        tsconfigRootDir: __dirname,
      },
    },
    settings: {
      "import/resolver": {
        typescript: {
          alwaysTryTypes: true,
          project: "./tsconfig.json",
        },
      },
    },
    rules: {
      // ===========================================
      // TYPESCRIPT STRICT RULES
      // ===========================================
      "@typescript-eslint/no-unused-vars": [
        "error",
        {
          argsIgnorePattern: "^_",
          varsIgnorePattern: "^_",
          caughtErrorsIgnorePattern: "^_",
        },
      ],
      "@typescript-eslint/no-explicit-any": "error",
      "@typescript-eslint/explicit-function-return-type": [
        "warn",
        {
          allowExpressions: true,
          allowTypedFunctionExpressions: true,
          allowHigherOrderFunctions: true,
        },
      ],
      "@typescript-eslint/prefer-nullish-coalescing": "error",
      "@typescript-eslint/prefer-optional-chain": "error",
      "@typescript-eslint/no-unnecessary-condition": "warn",
      "@typescript-eslint/no-unnecessary-type-assertion": "error",
      "@typescript-eslint/strict-boolean-expressions": [
        "warn",
        {
          allowString: false,
          allowNumber: false,
          allowNullableObject: false,
          allowNullableBoolean: false,
          allowNullableString: false,
          allowNullableNumber: false,
          allowAny: false,
        },
      ],

      // ===========================================
      // REACT 19 & HOOKS RULES
      // ===========================================
      "react-hooks/rules-of-hooks": "error",
      "react-hooks/exhaustive-deps": "error",
      "react/jsx-key": ["error", { checkFragmentShorthand: true }],
      "react/no-array-index-key": "warn",
      "react/jsx-no-leaked-render": "error",
      "react/jsx-curly-brace-presence": [
        "warn",
        { props: "never", children: "never" },
      ],
      "react/self-closing-comp": ["warn", { component: true, html: true }],
      "react/jsx-sort-props": [
        "warn",
        {
          callbacksLast: true,
          shorthandFirst: true,
          multiline: "last",
          reservedFirst: true,
        },
      ],

      // ===========================================
      // ACCESSIBILITY (WCAG 2.1 AA)
      // ===========================================
      "jsx-a11y/alt-text": "error",
      "jsx-a11y/aria-props": "error",
      "jsx-a11y/aria-proptypes": "error",
      "jsx-a11y/aria-unsupported-elements": "error",
      "jsx-a11y/role-has-required-aria-props": "error",
      "jsx-a11y/role-supports-aria-props": "error",
      "jsx-a11y/no-autofocus": ["error", { ignoreNonDOM: true }],
      "jsx-a11y/click-events-have-key-events": "error",
      "jsx-a11y/no-static-element-interactions": "error",
      "jsx-a11y/anchor-is-valid": [
        "error",
        {
          components: ["Link"],
          specialLink: ["hrefLeft", "hrefRight"],
          aspects: ["invalidHref", "preferButton"],
        },
      ],

      // ===========================================
      // IMPORT/EXPORT ORGANIZATION
      // ===========================================
      "import/order": [
        "error",
        {
          groups: [
            "builtin",
            "external",
            "internal",
            "parent",
            "sibling",
            "index",
            "object",
            "type",
          ],
          "newlines-between": "always",
          alphabetize: { order: "asc", caseInsensitive: true },
          pathGroups: [
            {
              pattern: "react",
              group: "external",
              position: "before",
            },
            {
              pattern: "next/**",
              group: "external",
              position: "before",
            },
            {
              pattern: "@/**",
              group: "internal",
              position: "before",
            },
          ],
          pathGroupsExcludedImportTypes: ["react", "next"],
        },
      ],
      "import/no-unused-modules": [
        "warn",
        { unusedExports: true, missingExports: true },
      ],
      "import/no-cycle": "error",
      "import/no-self-import": "error",

      // ===========================================
      // PERFORMANCE & BEST PRACTICES
      // ===========================================
      "no-console": ["warn", { allow: ["warn", "error"] }],
      "prefer-const": "error",
      "no-var": "error",
      "object-shorthand": "error",
      "prefer-template": "error",
      "prefer-arrow-callback": "error",
      "no-nested-ternary": "warn",
      "no-multiple-empty-lines": ["error", { max: 1, maxEOF: 0 }],
      "padding-line-between-statements": [
        "warn",
        { blankLine: "always", prev: "*", next: "return" },
        { blankLine: "always", prev: ["const", "let", "var"], next: "*" },
        {
          blankLine: "any",
          prev: ["const", "let", "var"],
          next: ["const", "let", "var"],
        },
      ],

      // ===========================================
      // NEXT.JS SPECIFIC OPTIMIZATIONS
      // ===========================================
      "@next/next/no-img-element": "error", // Use next/image instead
      "@next/next/no-html-link-for-pages": "error",
      "@next/next/no-sync-scripts": "error",
      "@next/next/no-css-tags": "error",

      // ===========================================
      // SECURITY & SAFETY
      // ===========================================
      "no-eval": "error",
      "no-implied-eval": "error",
      "no-new-func": "error",
      "no-script-url": "error",
      "no-alert": "warn",
      "no-debugger": "error",
    },
  },

  // ===========================================
  // FILE-SPECIFIC OVERRIDES
  // ===========================================
  {
    files: ["**/*.test.{js,jsx,ts,tsx}", "**/*.spec.{js,jsx,ts,tsx}"],
    rules: {
      "@typescript-eslint/no-explicit-any": "off",
      "@typescript-eslint/no-unsafe-assignment": "off",
      "@typescript-eslint/no-unsafe-call": "off",
      "@typescript-eslint/no-unsafe-member-access": "off",
      "no-console": "off",
    },
  },

  {
    files: ["**/*.config.{js,ts,mjs}", "**/*.setup.{js,ts}"],
    rules: {
      "import/no-default-export": "off",
      "@typescript-eslint/no-require-imports": "off",
    },
  },

  {
    files: ["app/**/*.{js,jsx,ts,tsx}"],
    rules: {
      // Next.js App Router specific rules
      "import/no-default-export": "off", // Page components need default exports
    },
  },

  {
    files: ["components/**/*.{js,jsx,ts,tsx}"],
    rules: {
      // Component-specific rules
      "react/jsx-sort-props": "off", // More flexibility for component props
      "@typescript-eslint/explicit-function-return-type": "off", // React components often infer return types
    },
  },

  // ===========================================
  // IGNORE PATTERNS
  // ===========================================
  {
    ignores: [
      "node_modules/**",
      ".next/**",
      "out/**",
      "coverage/**",
      "dist/**",
      "build/**",
      "*.d.ts",
      "public/**",
      ".env*",
    ],
  },
];

export default eslintConfig;
