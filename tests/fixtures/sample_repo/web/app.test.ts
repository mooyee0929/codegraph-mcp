import { start } from "./app";

test("start", () => {
  expect(start()).toBe(42);
});
