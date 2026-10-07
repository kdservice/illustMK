import type { IllustMKApi } from "../../preload/index";

declare global {
  interface Window {
    illustMK: IllustMKApi;
  }
}

export {};
