FROM nickgryg/alpine-pandas

RUN apk update; apk add jq

RUN mkdir /code
WORKDIR /code
RUN pip install requests tqdm

COPY collect.py ./

