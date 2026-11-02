from __future__ import annotations

from codegraph.services.clients import (
    detect_client_calls,
    graphql_client_calls,
    grpc_client_calls,
)
from codegraph.services.sources import SourceFile, language_of


def _source(path: str, text: str) -> SourceFile:
    labels = language_of(path)
    assert labels is not None
    return SourceFile(path=path, language=labels[0], family=labels[1], text=text)


def _calls(path: str, text: str) -> set[tuple[str, str, str]]:
    return {(c.method, c.path, c.library) for c in detect_client_calls(_source(path, text))}


def test_python_requests_httpx_and_sessions() -> None:
    found = _calls(
        "svc/client.py",
        "import requests\nBASE = 'http://users:8000'\nUSERS = '/users'\n"
        "def a(uid):\n"
        "    requests.get(f'{BASE}/users/{uid}')\n"
        "    httpx.post(BASE + USERS, json={})\n"
        "    requests.request('DELETE', f'{self.base}/users/{uid}')\n"
        "    self.session.put('/orders/' + oid)\n"
        "    cache.get('/not/a/call')\n"
        "    config.get('timeout')\n",
    )
    assert found == {
        ("GET", "/users/{uid}", "requests"),
        ("POST", "/users", "httpx"),
        ("DELETE", "/users/{uid}", "requests"),
        ("PUT", "/orders/{param}", "http-client"),
    }


def test_js_fetch_axios_and_instances() -> None:
    found = _calls(
        "web/api.ts",
        "const API = '/api';\n"
        "export async function load(id: string) {\n"
        "  await fetch(`${API}/users/${id}`);\n"
        "  await fetch('/api/users', { method: 'POST', body });\n"
        "  await axios.delete(`/api/users/${id}`);\n"
        "  await axios({ url: '/api/items', method: 'patch' });\n"
        "  await this.http.get<User[]>('/api/admins');\n"
        "  await apiClient.put('/api/users/' + id, data);\n"
        "  await fetch('/logo.png');\n"
        "  map.get('/x');\n"
        "}\n",
    )
    assert found == {
        ("GET", "/api/users/{id}", "fetch"),
        ("POST", "/api/users", "fetch"),
        ("DELETE", "/api/users/{id}", "axios"),
        ("PATCH", "/api/items", "axios"),
        ("GET", "/api/admins", "http-client"),
        ("PUT", "/api/users/{param}", "http-client"),
    }


def test_js_server_routes_are_not_client_calls() -> None:
    assert _calls("web/server.js", "app.get('/users', (req, res) => res.send([]));\n") == set()
    assert _calls("web/t.test.js", "await request(app).get('/users/1');\n") == {
        ("GET", "/users/1", "supertest")
    }


def test_go_clients() -> None:
    found = _calls(
        "client.go",
        'const base = "http://orders"\nfunc f() {\n'
        '\thttp.Get(base + "/orders/" + id)\n'
        '\treq, _ := http.NewRequest(http.MethodPost, "http://x/orders", body)\n'
        '\thttp.NewRequestWithContext(ctx, "DELETE", fmt.Sprintf("/orders/%d", id), nil)\n'
        '\tclient.R().SetBody(b).Put("/orders/1")\n}\n',
    )
    assert found == {
        ("GET", "/orders/{param}", "net/http"),
        ("POST", "/orders", "net/http"),
        ("DELETE", "/orders/{param}", "net/http"),
        ("PUT", "/orders/1", "resty"),
    }


def test_jvm_clients_including_declarative() -> None:
    found = _calls(
        "src/main/java/demo/Calls.java",
        "class Calls {\n"
        '  static final String USERS = "/users";\n'
        "  void a() {\n"
        '    restTemplate.getForObject("http://users" + USERS + "/{id}", User.class, id);\n'
        '    restTemplate.exchange("/users", HttpMethod.POST, entity, User.class);\n'
        '    webClient.get().uri("/users/{id}", id).retrieve();\n'
        "    HttpRequest r = HttpRequest.newBuilder()"
        '.uri(URI.create("http://x/items")).DELETE().build();\n'
        '    mockMvc.perform(get("/users/1"));\n'
        "  }\n}\n",
    )
    assert found == {
        ("GET", "/users/{id}", "resttemplate"),
        ("POST", "/users", "resttemplate"),
        ("GET", "/users/{id}", "webclient"),
        ("DELETE", "/items", "java-httpclient"),
        ("GET", "/users/1", "mockmvc"),
    }
    declared = detect_client_calls(
        _source(
            "Api.kt",
            'interface Api {\n  @GET("users/{id}")\n'
            '  suspend fun user(@Path("id") id: String): User\n}\n',
        )
    )
    assert [(c.method, c.path, c.decorated) for c in declared] == [("GET", "/users/{id}", "user")]
    feign = detect_client_calls(
        _source(
            "C.java",
            '@FeignClient("users")\ninterface C {\n'
            '  @PostMapping("/users")\n  User create(User u);\n}\n',
        )
    )
    assert [(c.method, c.library) for c in feign] == [("POST", "feign")]


def test_csharp_ruby_php_rust_clients() -> None:
    assert _calls(
        "Svc.cs",
        'var u = await _http.GetFromJsonAsync<User>($"api/users/{id}");\n'
        'var m = new HttpRequestMessage(HttpMethod.Delete, "/api/users/1");\n',
    ) == {("GET", "/api/users/{id}", "httpclient"), ("DELETE", "/api/users/1", "httpclient")}
    assert _calls(
        "c.rb",
        "HTTParty.get(\"https://x/users/#{id}\")\nFaraday.post '/orders', body\n",
    ) == {("GET", "/users/{id}", "HTTParty"), ("POST", "/orders", "Faraday")}
    assert _calls(
        "c.php",
        "<?php\n$r = Http::withToken($t)->get('https://x/api/users');\n"
        "$this->client->request('POST', '/api/orders');\n",
    ) == {("GET", "/api/users", "laravel-http"), ("POST", "/api/orders", "guzzle")}
    assert _calls(
        "src/c.rs",
        'let r = reqwest::get("http://x/health").await?;\nlet s = client.post("/items").send();\n',
    ) == {("GET", "/health", "reqwest"), ("POST", "/items", "reqwest")}


def test_grpc_stub_calls() -> None:
    go = _source(
        "cmd/c.go",
        "c := pb.NewGreeterClient(conn)\nr, err := c.SayHello(ctx, &pb.HelloRequest{})\n",
    )
    py = _source(
        "c.py",
        "stub = greeter_pb2_grpc.GreeterStub(channel)\nstub.SayHello(req)\nother.Unrelated()\n",
    )
    services = {"Greeter": ["SayHello", "StreamHellos"]}
    assert [c.path for c in grpc_client_calls(go, services)] == ["Greeter/SayHello"]
    assert [c.path for c in grpc_client_calls(py, services)] == ["Greeter/SayHello"]
    generated = _source("greeter_pb2_grpc.py", "GreeterStub(channel).SayHello(x)\n")
    assert grpc_client_calls(generated, services) == []


def test_graphql_operations() -> None:
    js = _source(
        "web/q.ts",
        "const Q = gql`\n  query Users($id: ID!) {\n    user(id: $id) { id ...F }\n"
        "    all: users { id }\n  }\n`;\nconst M = gql`mutation { addUser(input: {}) { id } }`;\n",
    )
    assert {c.path for c in graphql_client_calls(js)} == {
        "Query.user",
        "Query.users",
        "Mutation.addUser",
    }
    ops = _source("ops/get.graphql", "query GetUser { user(id: 1) { name } }\n")
    assert [c.path for c in graphql_client_calls(ops)] == ["Query.user"]
