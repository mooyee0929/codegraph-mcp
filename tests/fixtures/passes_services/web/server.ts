import express from "express";
import { Server } from "socket.io";

const app = express();
const io = new Server();

function health(req: unknown, res: { send(body: string): void }) {
  res.send("ok");
}

export function notify(orderId: number) {
  io.emit("order:updated", orderId);
}

io.on("order:updated", onUpdate);

function onUpdate(orderId: number) {
  return orderId;
}

app.get("/health", health);
app.post("/hooks/:source", (req, res) => res.send("ok"));
