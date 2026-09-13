import { helper } from "./util";

/** Handles a request. */
export class Handler {
  run(): number {
    return helper(this.compute());
  }
  compute(): number {
    return 41;
  }
}

export const start = () => new Handler().run();
