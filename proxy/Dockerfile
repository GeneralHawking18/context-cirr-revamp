# proxy/Dockerfile
FROM alpine:3.19

RUN apk add --no-cache tinyproxy ca-certificates && update-ca-certificates

# Copy config
COPY tinyproxy.conf /etc/tinyproxy/tinyproxy.conf
COPY allowed-domains /etc/tinyproxy/allowed-domains

EXPOSE 8888
CMD ["/usr/bin/tinyproxy", "-d", "-c", "/etc/tinyproxy/tinyproxy.conf"]