import logging
import hashlib
import base64

from sqlalchemy import Table, Column, Integer, String, LargeBinary, Text, MetaData, Date, ARRAY, create_engine
from sqlalchemy.exc import IntegrityError, ProgrammingError


def get_logger(name: str):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(levelname)s %(message)s')
    return logging.getLogger(name)


logger = get_logger(__name__)

PG_CONN = 'postgresql://zh:pwd@localhost:5732/news2image'
db = create_engine(PG_CONN)
metadata = MetaData(db)

logs = Table('logs', metadata,
             Column('id', String, primary_key=True),
             Column('news_text', Text),
             Column('news_summary', Text),
             Column('news_keywords', ARRAY(String)),
             Column('image_captions', ARRAY(String)),
             Column('image_keywords', ARRAY(String)),
             Column('collage', LargeBinary)
             )

conn = db.connect()

try:
    logs.create()
except ProgrammingError:
    pass

t_txt = """Oil prices crashed in Asia on Monday by around 30% in what analysts are calling the start of a price war.
    Top oil exporter Saudi Arabia slashed its oil prices at the weekend after it failed to convince Russia on Friday to back sharp production cuts.
    Oil cartel Opec and its ally Russia had previously worked together on production curbs.
    The benchmark Brent oil futures plunged to a low of $31.02 a barrel on Monday, in volatile energy markets.
    Oil prices have now fallen 30% since Friday, when Opec's 14 members led by Saudi Arabia met with its allies Russia and other non-Opec members.
    They met to discuss how to respond to falling demand caused by the growing spread of the coronavirus."""


def get_id(text: str) -> str:
    return hashlib.sha3_256(text.encode()).hexdigest()


def insert_log(text: str):
    insert_statement = logs.insert().values(id=get_id(text=text), news_text=text)
    try:
        conn.execute(insert_statement)
    except IntegrityError as ie:
        logger.warning(f"Got PG Integrity error while saving items: {ie}")


def update_summary(text: str, summary: str):
    _id = get_id(text)
    update_statement = logs.update().where(logs.c.id==_id).values(news_summary=summary)
    try:
        conn.execute(update_statement)
    except IntegrityError as ie:
        logger.warning(f"Got PG Integrity error while saving items: {ie}")


def update_news_keywords(text: str, keywords: set):
    _id = get_id(text)
    update_statement = logs.update().where(logs.c.id == _id).values(news_keywords=keywords)
    try:
        conn.execute(update_statement)
    except IntegrityError as ie:
        logger.warning(f"Got PG Integrity error while saving items: {ie}")


def update_image_captions(text: str, captions: list):
    _id = get_id(text)
    update_statement = logs.update().where(logs.c.id == _id).values(image_captions=captions)
    try:
        conn.execute(update_statement)
    except IntegrityError as ie:
        logger.warning(f"Got PG Integrity error while saving items: {ie}")


def update_image_keywords(text: str, keywords: set):
    _id = get_id(text)
    update_statement = logs.update().where(logs.c.id == _id).values(image_keywords=keywords)
    try:
        conn.execute(update_statement)
    except IntegrityError as ie:
        logger.warning(f"Got PG Integrity error while saving items: {ie}")


def update_collage(text: str, image_b64: bytes):
    _id = get_id(text)
    update_statement = logs.update().where(logs.c.id == _id).values(collage=image_b64)
    try:
        conn.execute(update_statement)
    except IntegrityError as ie:
        logger.warning(f"Got PG Integrity error while saving items: {ie}")


def select_image(text: str) -> bytes:
    _id = get_id(text)
    select_statement = logs.select().where(logs.c.id==_id)
    try:
        res = conn.execute(select_statement)
        for r in res:
            print(r)
    except IntegrityError as ie:
        logger.warning(f"Got PG Integrity error while saving items: {ie}")

# insert_log(text=t_txt)
update_summary(text=t_txt, summary="ds")
update_news_keywords(text=t_txt, keywords={'key', 'word'})
update_image_captions(text=t_txt, captions=["asa", "aswer"])
update_image_keywords(text=t_txt, keywords={'1', '33'})

with open('../../data/images/goi5k/3741949_aef5340c10_o.jpg', 'rb') as f:
    data = base64.b64encode(f.read())
print(type(data))
# update_collage(text=t_txt, image_b64=data)
select_image(text=t_txt)
