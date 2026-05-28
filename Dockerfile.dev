FROM python:3.14-alpine AS executor

WORKDIR /app

RUN apk --no-cache add \
      envsubst \
      shadow \
      uv \
      bash
RUN useradd -m admin

COPY requirements.txt /tmp/requirements.txt
RUN uv pip install --system -r /tmp/requirements.txt

ENV RUN_CMD="bash"

ENTRYPOINT ["sh", "-c", "\
    if [ $0 != sh ] || [ $# -gt 0 ];then \
        export RUN_CMD=\"$0 $@\"; \
    fi; \
    if [ $(stat -c '%u' /app) -eq 0 ]; then \
        ${RUN_CMD}; \
    else \
        groupmod -g $(stat -c '%u' /app) admin; \
        usermod -u $(stat -c '%u' /app) -g $(stat -c '%u' /app) admin; \
        ln -s /app/.bash_history /home/admin/.bash_history; \
        chown admin:admin /home/admin; \
        su admin -c '${RUN_CMD}'; \
    fi \
    "]
CMD []
