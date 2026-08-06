# Dockerfile
#
# Multi-stage build for the Next.js frontend. Production image only
# contains the built output + a minimal Node runtime, not the full
# node_modules dev toolchain.

FROM node:20-slim AS builder
WORKDIR /app

COPY package.json package-lock.json ./
RUN npm ci

COPY . .
# Baked in at build time — Next.js needs NEXT_PUBLIC_* vars available
# when the app is built, not just at container runtime.
ARG NEXT_PUBLIC_API_BASE_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_BASE_URL=$NEXT_PUBLIC_API_BASE_URL
RUN npm run build

FROM node:20-slim AS runtime
WORKDIR /app
RUN useradd --create-home --uid 1000 appuser

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

USER appuser
EXPOSE 3000
ENV PORT=3000

CMD ["node", "server.js"]
