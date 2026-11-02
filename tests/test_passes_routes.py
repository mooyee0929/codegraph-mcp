from __future__ import annotations

from codegraph.services.paths import (
    compile_path,
    join_paths,
    match_route,
    normalize_method,
    normalize_path,
    split_route_qname,
)
from codegraph.services.route_detectors import handler_ref, resolve_js_specifier
from codegraph.services.routes import RouteDef, detect_routes
from codegraph.services.sources import SourceFile, language_of


def _source(path: str, text: str) -> SourceFile:
    labels = language_of(path)
    assert labels is not None, path
    return SourceFile(path=path, language=labels[0], family=labels[1], text=text)


def _defs(*files: tuple[str, str]) -> list[RouteDef]:
    return detect_routes([_source(p, t) for p, t in files])


def _routes(*files: tuple[str, str]) -> set[str]:
    return {r.qname.removeprefix("route:") for r in _defs(*files)}


# -- paths ----------------------------------------------------------------------


def test_normalize_path_folds_parameter_spellings() -> None:
    assert normalize_path("/users/:id") == "/users/{id}"
    assert normalize_path("/users/<int:user_id>/") == "/users/{user_id}"
    assert normalize_path("users/{id:int}/posts/{slug?}") == "/users/{id}/posts/{slug}"
    assert normalize_path("^articles/(?P<year>[0-9]{4})/$") == "/articles/{year}"
    assert normalize_path("https://api.example.com:8443/v1/items?x=1") == "/v1/items"
    assert normalize_path("/users/${user.id}/orders") == "/users/{id}/orders"
    assert normalize_path("/files/[...slug]") == "/files/{*}"
    assert normalize_path("/a/%s/b") == "/a/{param}/b"
    assert normalize_path("/photos(.:format)") == "/photos"
    assert normalize_path("") == "/"


def test_normalize_method_and_join() -> None:
    assert normalize_method("http.MethodPost") == "POST"
    assert normalize_method("RequestMethod.GET") == "GET"
    assert normalize_method("all") == "ANY"
    assert normalize_method(None) == "ANY"
    assert join_paths("/api/", "", "users", "/{id}") == "/api/users/{id}"
    assert split_route_qname("route:GET /a/b") == ("GET", "/a/b")
    assert split_route_qname("channel:x") is None


def test_match_route_scores_parameters_and_prefixes() -> None:
    routes = [
        compile_path("GET", "/users/{id}"),
        compile_path("GET", "/users/me"),
        compile_path("POST", "/users"),
        compile_path("ANY", "/health"),
    ]
    exact = match_route("GET", "/users/me", routes)
    assert exact is not None and exact.qname == "route:GET /users/me" and exact.score == 1.0
    param = match_route("GET", "/users/42", routes)
    assert param is not None and param.qname == "route:GET /users/{id}"
    unknown = match_route("GET", "/users/{param}", routes)
    assert unknown is not None and unknown.qname == "route:GET /users/{id}"
    assert match_route("DELETE", "/users", routes) is None
    gateway = match_route("POST", "/api/v1/users", routes)
    assert gateway is not None and gateway.qname == "route:POST /users" and gateway.score < 1.0
    any_method = match_route("GET", "/health", routes)
    assert any_method is not None and any_method.score == 0.95


def test_handler_ref_and_js_specifiers() -> None:
    assert handler_ref("views.user_detail", "python") == "views.user_detail"
    assert handler_ref("UserView.as_view()", "python") == "UserView"
    assert handler_ref("http.HandlerFunc(listUsers)", "go") == "listUsers"
    assert handler_ref("(req, res) => res.send()", "js") is None
    assert handler_ref("func(c *gin.Context) {}", "go") is None
    assert "web/routes/users.ts" in resolve_js_specifier("web/app.ts", "./routes/users")
    assert "web/routes/index.ts" in resolve_js_specifier("web/app.ts", "./routes")
    assert resolve_js_specifier("web/app.ts", "express") == []


# -- Python -----------------------------------------------------------------------


def test_flask_routes_and_blueprint_prefixes() -> None:
    found = _routes(
        (
            "app/views.py",
            "from flask import Flask, Blueprint\n"
            "app = Flask(__name__)\n"
            "bp = Blueprint('users', __name__, url_prefix='/users')\n"
            "@app.route('/')\ndef index():\n    return ''\n"
            "@bp.route('/<int:uid>', methods=['GET', 'DELETE'])\ndef user(uid):\n    return ''\n"
            "@app.post('/login')\ndef login():\n    return ''\n"
            "app.add_url_rule('/ping', view_func=ping)\n",
        )
    )
    assert found == {
        "GET /",
        "GET /users/{uid}",
        "DELETE /users/{uid}",
        "POST /login",
        "GET /ping",
    }


def test_fastapi_router_prefix_and_cross_file_include() -> None:
    found = _routes(
        (
            "svc/routers/items.py",
            "from fastapi import APIRouter\n"
            "router = APIRouter(prefix='/items')\n"
            "@router.get('/{item_id}')\nasync def read_item(item_id: int):\n    return {}\n"
            "@router.api_route('/bulk', methods=['PUT', 'PATCH'])\ndef bulk():\n    return {}\n",
        ),
        (
            "svc/main.py",
            "from fastapi import FastAPI\nfrom svc.routers import items\n"
            "app = FastAPI()\napp.include_router(items.router, prefix='/api')\n"
            "@app.get('/health')\ndef health():\n    return {}\n",
        ),
    )
    assert found == {
        "GET /api/items/{item_id}",
        "PUT /api/items/bulk",
        "PATCH /api/items/bulk",
        "GET /health",
    }


def test_django_urls_with_include() -> None:
    found = _routes(
        (
            "site/urls.py",
            "from django.urls import include, path\n"
            "urlpatterns = [\n"
            "    path('blog/', include('blog.urls')),\n"
            "    path('', views.home, name='home'),\n"
            "]\n",
        ),
        (
            "blog/urls.py",
            "from django.urls import path, re_path\nfrom . import views\n"
            "urlpatterns = [\n"
            "    path('<slug:slug>/', views.detail),\n"
            "    re_path(r'^archive/(?P<year>[0-9]{4})/$', views.ArchiveView.as_view()),\n"
            "]\n",
        ),
    )
    assert found == {"ANY /", "ANY /blog/{slug}", "ANY /blog/archive/{year}"}
    detail = next(d for d in _defs(("blog/urls.py", "urlpatterns = [path('x/', views.detail)]")))
    assert detail.handler == "views.detail" and detail.framework == "django"


def test_starlette_routes_inside_mounts() -> None:
    found = _routes(
        (
            "asgi.py",
            "from starlette.applications import Starlette\n"
            "from starlette.routing import Mount, Route\n"
            "api = [Route('/users', list_users), Route('/users/{id}', get_user, methods=['GET'])]\n"
            "app = Starlette(routes=[Route('/', homepage), Mount('/api', routes=api)])\n",
        )
    )
    assert found == {"GET /", "GET /api/users", "GET /api/users/{id}"}


# -- JavaScript / TypeScript --------------------------------------------------------


def test_express_router_mounted_from_another_file() -> None:
    found = _routes(
        (
            "web/routes/users.ts",
            "import express from 'express';\n"
            "const router = express.Router();\n"
            "router.get('/:id', getUser);\n"
            "router.post('/', auth, async (req, res) => res.send());\n"
            "router.route('/:id/avatar').get(getAvatar).put(setAvatar);\n"
            "export default router;\n",
        ),
        (
            "web/app.ts",
            "import express from 'express';\nimport users from './routes/users';\n"
            "const app = express();\napp.use('/api/users', users);\n"
            "app.get('/health', (req, res) => res.send('ok'));\n"
            "cache.get('/not-a-route');\n",
        ),
    )
    assert found == {
        "GET /api/users/{id}",
        "POST /api/users",
        "GET /api/users/{id}/avatar",
        "PUT /api/users/{id}/avatar",
        "GET /health",
    }


def test_koa_fastify_and_hapi() -> None:
    assert _routes(
        (
            "k.js",
            "const Router = require('@koa/router');\n"
            "const router = new Router({ prefix: '/v2' });\n"
            "router.get('/things', listThings);\n",
        )
    ) == {"GET /v2/things"}
    assert _routes(
        (
            "f.js",
            "const fastify = require('fastify')();\n"
            "fastify.get('/a', async () => 1);\n"
            "fastify.route({ method: ['POST', 'PUT'], url: '/b', handler: saveB });\n",
        )
    ) == {"GET /a", "POST /b", "PUT /b"}
    assert _routes(
        ("h.js", "server.route({ method: 'GET', path: '/hapi/{id}', handler: (r, h) => 1 });\n")
    ) == {"GET /hapi/{id}"}


def test_nestjs_controller_with_global_prefix() -> None:
    defs = _defs(
        (
            "src/users.controller.ts",
            "import { Controller, Get, Post } from '@nestjs/common';\n"
            "@Controller('users')\n"
            "export class UsersController {\n"
            "  @Get(':id')\n  findOne(@Param('id') id: string) { return id; }\n"
            "  @Post()\n  async create(@Body() dto: Dto) { return dto; }\n"
            "}\n",
        ),
        (
            "src/main.ts",
            "const app = await NestFactory.create(AppModule);\napp.setGlobalPrefix('api');\n",
        ),
    )
    assert {d.qname for d in defs} == {"route:GET /api/users/{id}", "route:POST /api/users"}
    assert {d.handler for d in defs} == {"findOne", "create"}


def test_nextjs_and_sveltekit_file_routes() -> None:
    found = _routes(
        ("pages/api/users/[id].ts", "export default function handler(req, res) {}\n"),
        (
            "src/app/(shop)/orders/[orderId]/route.ts",
            "export async function GET() {}\nexport const POST = () => 1;\n",
        ),
        ("src/routes/items/[slug]/+server.ts", "export function DELETE() {}\n"),
        ("app/_private/route.ts", "export function GET() {}\n"),
    )
    assert found == {
        "ANY /api/users/{id}",
        "GET /orders/{orderId}",
        "POST /orders/{orderId}",
        "DELETE /items/{slug}",
    }


# -- Go ---------------------------------------------------------------------------


def test_go_net_http_gin_echo_chi_gorilla() -> None:
    assert _routes(
        (
            "main.go",
            'package main\nimport "net/http"\nfunc main() {\n'
            '\thttp.HandleFunc("/hello", hello)\n'
            '\tmux.HandleFunc("GET /users/{id}", getUser)\n}\n',
        )
    ) == {"ANY /hello", "GET /users/{id}"}
    assert _routes(
        (
            "gin.go",
            'import "github.com/gin-gonic/gin"\nfunc r() {\n\tr := gin.Default()\n'
            '\tv1 := r.Group("/v1")\n\tv1.GET("/users/:id", handlers.GetUser)\n'
            '\tr.POST("/login", func(c *gin.Context) {})\n}\n',
        )
    ) == {"GET /v1/users/{id}", "POST /login"}
    assert _routes(
        (
            "echo.go",
            'import "github.com/labstack/echo/v4"\nfunc r() {\n\te := echo.New()\n'
            '\tg := e.Group("/admin")\n\tg.DELETE("/users/:id", deleteUser)\n}\n',
        )
    ) == {"DELETE /admin/users/{id}"}
    assert _routes(
        (
            "chi.go",
            'import "github.com/go-chi/chi/v5"\nfunc r() {\n\tr := chi.NewRouter()\n'
            '\tr.Route("/articles", func(r chi.Router) {\n'
            '\t\tr.Get("/{articleID}", getArticle)\n\t})\n'
            '\tr.Mount("/admin", adminRouter())\n}\n'
            "func adminRouter() http.Handler {\n\tr := chi.NewRouter()\n"
            '\tr.Get("/stats", stats)\n\treturn r\n}\n',
        )
    ) == {"GET /articles/{articleID}", "GET /admin/stats"}
    assert _routes(
        (
            "mux.go",
            'import "github.com/gorilla/mux"\nfunc r() {\n\tr := mux.NewRouter()\n'
            '\ts := r.PathPrefix("/api").Subrouter()\n'
            '\ts.HandleFunc("/products/{key}", product).Methods("GET", "PUT")\n}\n',
        )
    ) == {"GET /api/products/{key}", "PUT /api/products/{key}"}


# -- JVM --------------------------------------------------------------------------


def test_spring_class_prefix_and_feign_skipped() -> None:
    defs = _defs(
        (
            "src/main/java/demo/UserController.java",
            '@RestController\n@RequestMapping("/api/users")\npublic class UserController {\n'
            '  @GetMapping("/{id}")\n  public User get(@PathVariable Long id) { return null; }\n'
            "  @PostMapping\n"
            "  public ResponseEntity<User> create(@RequestBody User u) { return null; }\n"
            '  @RequestMapping(value = {"/a", "/b"}, method = RequestMethod.PUT)\n'
            "  public void both() {}\n"
            "}\n",
        ),
        (
            "src/main/java/demo/Client.java",
            '@FeignClient(name = "users")\npublic interface Client {\n'
            '  @GetMapping("/remote")\n  User remote();\n}\n',
        ),
    )
    assert {d.qname for d in defs} == {
        "route:GET /api/users/{id}",
        "route:POST /api/users",
        "route:PUT /api/users/a",
        "route:PUT /api/users/b",
    }
    assert {d.handler for d in defs} == {"get", "create", "both"}


def test_jaxrs_micronaut_and_ktor() -> None:
    assert _routes(
        (
            "R.java",
            '@Path("/orders")\npublic class OrderResource {\n'
            '  @GET\n  @Path("{id}")\n  @Produces("application/json")\n'
            '  public Order find(@PathParam("id") String id) { return null; }\n'
            "  @POST\n  public void add(Order o) {}\n}\n",
        )
    ) == {"GET /orders/{id}", "POST /orders"}
    assert _routes(
        (
            "Hello.kt",
            '@Controller("/hello")\nclass HelloController {\n'
            '    @Get("/{name}")\n    fun greet(name: String): String = name\n}\n',
        )
    ) == {"GET /hello/{name}"}
    assert _routes(
        (
            "App.kt",
            'fun Application.module() {\n  routing {\n    route("/api") {\n'
            '      get("/items") { call.respond(1) }\n'
            "      post { call.respond(2) }\n    }\n  }\n}\n",
        )
    ) == {"GET /api/items", "POST /api"}


# -- C# ---------------------------------------------------------------------------


def test_aspnet_attribute_and_minimal_routes() -> None:
    defs = _defs(
        (
            "Controllers/UsersController.cs",
            '[ApiController]\n[Route("api/[controller]")]\n'
            "public class UsersController : ControllerBase {\n"
            '    [HttpGet("{id:int}")]\n'
            "    public async Task<ActionResult<User>> GetUser(int id) => null;\n"
            "    [HttpPost]\n    public IActionResult Create(User u) => Ok();\n"
            '    [HttpGet("/health")]\n    public IActionResult Health() => Ok();\n}\n',
        ),
        (
            "Program.cs",
            'var app = builder.Build();\napp.MapGet("/todos/{id}", (int id) => id);\n'
            'var g = app.MapGroup("/admin");\ng.MapDelete("/cache", Handlers.Clear);\n',
        ),
    )
    assert {d.qname for d in defs} == {
        "route:GET /api/users/{id}",
        "route:POST /api/users",
        "route:GET /health",
        "route:GET /todos/{id}",
        "route:DELETE /admin/cache",
    }


# -- Ruby -------------------------------------------------------------------------


def test_rails_routes_resources_namespaces_and_verbs() -> None:
    defs = _defs(
        (
            "config/routes.rb",
            "Rails.application.routes.draw do\n"
            "  root 'pages#home'\n"
            "  get '/about', to: 'pages#about'\n"
            "  resources :photos, only: [:index, :show] do\n"
            "    resources :comments, only: [:create]\n"
            "    member do\n      post :publish\n    end\n"
            "  end\n"
            "  namespace :admin do\n"
            "    resources :users, except: [:new, :edit, :create, :update, :destroy, :index]\n"
            "  end\n"
            "  resource :profile, only: [:show]\n"
            "  match 'search', to: 'search#run', via: [:get, :post]\n"
            "end\n",
        )
    )
    assert {d.qname.removeprefix("route:") for d in defs} == {
        "GET /",
        "GET /about",
        "GET /photos",
        "GET /photos/{id}",
        "POST /photos/{photo_id}/comments",
        "POST /photos/{id}/publish",
        "GET /admin/users/{id}",
        "GET /profile",
        "GET /search",
        "POST /search",
    }
    show = next(d for d in defs if d.qname == "route:GET /photos/{id}")
    assert (show.handler, show.owner_hint) == ("show", "PhotosController")
    admin = next(d for d in defs if d.qname == "route:GET /admin/users/{id}")
    assert admin.owner_hint == "UsersController"


def test_sinatra_routes() -> None:
    assert _routes(
        ("app.rb", "require 'sinatra'\nget '/hi/:name' do\n  'hi'\nend\npost('/x') { 1 }\n")
    ) == {"GET /hi/{name}", "POST /x"}


# -- PHP --------------------------------------------------------------------------


def test_laravel_routes_groups_and_resources() -> None:
    defs = _defs(
        (
            "routes/api.php",
            "<?php\nRoute::get('/users/{id?}', [UserController::class, 'show']);\n"
            "Route::prefix('admin')->group(function () {\n"
            "    Route::post('/flush', 'CacheController@flush');\n"
            "    Route::apiResource('photos', PhotoController::class)->only(['index', 'show']);\n"
            "});\n"
            "Route::match(['get', 'post'], '/search', SearchController::class);\n",
        )
    )
    assert {d.qname.removeprefix("route:") for d in defs} == {
        "GET /api/users/{id}",
        "POST /api/admin/flush",
        "GET /api/admin/photos",
        "GET /api/admin/photos/{photo}",
        "GET /api/search",
        "POST /api/search",
    }
    show = next(d for d in defs if d.qname == "route:GET /api/users/{id}")
    assert (show.handler, show.owner_hint) == ("show", "UserController")


def test_symfony_route_attributes() -> None:
    assert _routes(
        (
            "src/Controller/BlogController.php",
            "<?php\n#[Route('/blog')]\nclass BlogController {\n"
            "    #[Route('/{slug}', name: 'post', methods: ['GET'])]\n"
            "    public function show(string $slug) {}\n}\n",
        )
    ) == {"GET /blog/{slug}"}


# -- Rust -------------------------------------------------------------------------


def test_actix_rocket_axum_routes() -> None:
    assert _routes(
        (
            "src/a.rs",
            'use actix_web::get;\n#[get("/users/{id}")]\nasync fn user(id: web::Path<u32>) {}\n',
        )
    ) == {"GET /users/{id}"}
    assert _routes(
        (
            "src/b.rs",
            'fn app() -> Router {\n    Router::new()\n        .route("/", get(root))\n'
            '        .route("/users", post(create_user).get(list_users))\n'
            '        .nest("/api", api_routes())\n}\n'
            'fn api_routes() -> Router {\n    Router::new().route("/items/:id", get(item))\n}\n',
        )
    ) == {"GET /", "POST /users", "GET /users", "GET /api/items/{id}"}


# -- schemas ------------------------------------------------------------------------


def test_grpc_services_from_proto() -> None:
    defs = _defs(
        (
            "proto/greeter.proto",
            'syntax = "proto3";\nservice Greeter {\n'
            "  // rpc Hidden (A) returns (B);\n"
            "  rpc SayHello (HelloRequest) returns (HelloReply);\n"
            "  rpc StreamHellos (stream HelloRequest) returns (stream HelloReply) {}\n}\n",
        )
    )
    assert {d.qname for d in defs} == {
        "route:GRPC Greeter/SayHello",
        "route:GRPC Greeter/StreamHellos",
    }


def test_graphql_root_fields_from_sdl_and_templates() -> None:
    found = _routes(
        (
            "schema.graphql",
            "schema { query: RootQuery mutation: Mutation }\n"
            'type RootQuery {\n  "doc"\n'
            "  user(id: ID!, filter: Filter): User\n  users: [User!]!\n}\n"
            'type Mutation { addUser(input: NewUser): User @deprecated(reason: "x") }\n'
            "type User { id: ID! }\n",
        ),
        ("server.ts", "const typeDefs = gql`\n  extend type Query {\n    ping: String\n  }\n`;\n"),
    )
    assert found == {
        "GRAPHQL Query.user",
        "GRAPHQL Query.users",
        "GRAPHQL Mutation.addUser",
        "GRAPHQL Query.ping",
    }


def test_routes_in_test_files_and_comments_are_ignored() -> None:
    assert (
        _routes(
            ("tests/test_app.py", "@app.get('/fake')\ndef fake():\n    pass\n"),
            ("web/app.ts", "// app.get('/commented', handler)\n"),
        )
        == set()
    )
