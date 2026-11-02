# Next.js web client, production build. Build from this folder (docker-compose.yml does this):
#   docker build -t brain-tumour-web .
FROM node:22-alpine AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
# public/ holds only the gitignored render, so it may not exist in a fresh clone.
RUN mkdir -p public
# The browser calls the API directly, so its address is fixed into the client bundle at build time.
ARG NEXT_PUBLIC_API_URL=http://localhost:8010
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL} \
    NEXT_TELEMETRY_DISABLED=1
RUN npm run build

FROM node:22-alpine
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000
COPY --from=build --chown=node:node /app/.next/standalone ./
COPY --from=build --chown=node:node /app/.next/static ./.next/static
# public/generated/ (the Overview's patient-derived render) is mounted at run time, not copied.
COPY --from=build --chown=node:node /app/public ./public
USER node
EXPOSE 3000
CMD ["node", "server.js"]
