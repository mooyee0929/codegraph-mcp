const API = "/api";

export async function loadOrder(id: number) {
  const res = await fetch(`${API}/orders/${id}`);
  return res.json();
}

export async function createOrder() {
  return fetch(`${API}/orders`, { method: "POST" });
}

export async function loadInvoices() {
  return fetch("/billing/invoices");
}
