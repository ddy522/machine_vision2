import sqlite3
import uuid

# DB 연결 (파일명 example.db)
conn = sqlite3.connect('vision.db')
cur = conn.cursor()

# 테이블 생성
cur.executescript("""
CREATE TABLE IF NOT EXISTS bomh (
    skey TEXT PRIMARY KEY,
    part_code TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bom (
    pskey TEXT NOT NULL,
    part_code VARCHAR(10) NOT NULL,
    part_name VARCHAR(20) NOT NULL,
    part_seq INTEGER NOT NULL,
    useage INTEGER NOT NULL,
    FOREIGN KEY (pskey) REFERENCES bomh(skey)
);

CREATE TABLE IF NOT EXISTS bom_cnt (
    part_code VARCHAR(10) NOT NULL,
    input_useage INTEGER NOT NULL,
    lotno VARCHAR(11) NOT NULL
);
""")

# 기본 데이터 삽입 (중복 삽입 방지 위해 체크)
def insert_if_not_exists(table, key_field, key_value, data_tuple):
    cur.execute(f"SELECT 1 FROM {table} WHERE {key_field} = ?", (key_value,))
    if cur.fetchone() is None:
        placeholders = ','.join(['?'] * len(data_tuple))
        cur.execute(f"INSERT INTO {table} VALUES ({placeholders})", data_tuple)

# bomh 삽입
product1_skey = str(uuid.uuid4())
insert_if_not_exists('bomh', 'skey', product1_skey, (product1_skey, 'Product1'))

# bom 삽입
bom_data = [
    (product1_skey, 'red_rec', '빨간네모', 1, 1),
    (product1_skey, 'green_rec', '초록네모', 2, 1),
    (product1_skey, 'blue_tri', '파란세모', 3, 2)
]
for row in bom_data:
    insert_if_not_exists('bom', 'part_code', row[0], row)

# bom_cnt 예시 데이터
bom_cnt_data = [
    ('red_rec', 10, 'LOT00000001'),
    ('green_rec', 5, 'LOT00000002'),
    ('blue_tri', 8, 'LOT00000003')
]
for row in bom_cnt_data:
    # key가 없으니 그냥 insert
    cur.execute("INSERT INTO bom_cnt (part_code, input_useage, lotno) VALUES (?, ?, ?)", row)

conn.commit()
conn.close()
