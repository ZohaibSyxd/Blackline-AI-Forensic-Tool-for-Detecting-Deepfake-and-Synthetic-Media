/// <reference types="vite/client" />

// Extend ImportMetaEnv for our custom vars
interface ImportMetaEnv {
  readonly VITE_API_BASE?: string;
}
interface ImportMeta {
  readonly env: ImportMetaEnv;
}
