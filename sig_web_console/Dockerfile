# syntax=docker/dockerfile:1
# The web console (docs/08_interface.md): nginx serving the static UI and routing /api/ to the two services.
# Build context: the repository root.

FROM nginxinc/nginx-unprivileged:1.30.5-alpine
COPY sig_web_console/nginx.conf /etc/nginx/conf.d/default.conf
COPY sig_web_console/static /usr/share/nginx/html
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=5s --start-period=5s --retries=3 CMD ["wget", "-qO", "/dev/null", "http://127.0.0.1:8080/"]
