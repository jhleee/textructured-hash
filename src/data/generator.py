"""Synthetic dataset generator"""

import random
import string
import json
from typing import List, Dict, Tuple
from datetime import datetime, timedelta
import hashlib
import uuid
import base64


class DatasetGenerator:
    """합성 데이터셋 생성기"""

    def __init__(self, seed: int = 42):
        random.seed(seed)
        self.seed = seed

    def generate_url(self, count: int = 1000) -> List[Dict]:
        """URL 생성"""
        domains = ['example', 'test', 'demo', 'site', 'web', 'app', 'api', 'service']
        tlds = ['com', 'net', 'org', 'io', 'dev', 'ai', 'co', 'kr']
        protocols = ['https', 'http', 'ftp']
        paths = ['users', 'api', 'docs', 'data', 'files', 'images', 'posts', 'items']

        samples = []
        for i in range(count):
            protocol = random.choice(protocols)
            domain = random.choice(domains) + str(random.randint(1, 999))
            tld = random.choice(tlds)

            if random.random() < 0.6:
                # With path
                path = '/'.join(random.sample(paths, random.randint(1, 3)))
                url = f"{protocol}://{domain}.{tld}/{path}"
                if random.random() < 0.3:
                    # Add query
                    url += f"?id={random.randint(1, 9999)}&page={random.randint(1, 10)}"
            else:
                # Simple URL
                url = f"{protocol}://{domain}.{tld}"

            samples.append({'text': url, 'category': 'url'})
        return samples

    def generate_email(self, count: int = 1000) -> List[Dict]:
        """이메일 생성"""
        names = ['john', 'jane', 'alice', 'bob', 'charlie', 'david', 'emma', 'frank']
        surnames = ['smith', 'jones', 'brown', 'wilson', 'taylor', 'lee', 'kim', 'park']
        domains = ['gmail', 'yahoo', 'hotmail', 'outlook', 'company', 'work', 'test']
        tlds = ['com', 'net', 'org', 'co.kr', 'io']

        samples = []
        for i in range(count):
            if random.random() < 0.5:
                # Simple format
                name = random.choice(names) + str(random.randint(1, 999))
                email = f"{name}@{random.choice(domains)}.{random.choice(tlds)}"
            else:
                # First.Last format
                first = random.choice(names)
                last = random.choice(surnames)
                email = f"{first}.{last}@{random.choice(domains)}.{random.choice(tlds)}"

            samples.append({'text': email, 'category': 'email'})
        return samples

    def generate_phone(self, count: int = 500) -> List[Dict]:
        """전화번호 생성"""
        samples = []
        formats = [
            lambda: f"010-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
            lambda: f"+82-10-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}",
            lambda: f"({random.randint(100, 999)}) {random.randint(100, 999)}-{random.randint(1000, 9999)}",
            lambda: f"{random.randint(100, 999)}-{random.randint(100, 999)}-{random.randint(1000, 9999)}"
        ]

        for i in range(count):
            phone = random.choice(formats)()
            samples.append({'text': phone, 'category': 'phone'})
        return samples

    def generate_date(self, count: int = 500) -> List[Dict]:
        """날짜 생성"""
        samples = []
        base_date = datetime(2020, 1, 1)

        for i in range(count):
            date = base_date + timedelta(days=random.randint(0, 1500))
            format_choice = random.randint(0, 3)

            if format_choice == 0:
                text = date.strftime("%Y-%m-%d")
            elif format_choice == 1:
                text = date.strftime("%d/%m/%Y")
            elif format_choice == 2:
                text = date.strftime("%m/%d/%Y")
            else:
                text = f"{date.year}년 {date.month}월 {date.day}일"

            samples.append({'text': text, 'category': 'date'})
        return samples

    def generate_ipv4(self, count: int = 300) -> List[Dict]:
        """IPv4 주소 생성"""
        samples = []
        for i in range(count):
            ip = f"{random.randint(1, 255)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 255)}"
            samples.append({'text': ip, 'category': 'ipv4'})
        return samples

    def generate_json(self, count: int = 1000) -> List[Dict]:
        """JSON 생성"""
        samples = []
        keys = ['name', 'id', 'value', 'status', 'type', 'data', 'user', 'item', 'count', 'price']
        values = ['active', 'pending', 'complete', 'success', 'error', 'true', 'false']

        for i in range(count):
            depth = random.randint(1, 3)
            num_keys = random.randint(2, 8)

            obj = {}
            for _ in range(num_keys):
                key = random.choice(keys)
                if depth > 1 and random.random() < 0.3:
                    # Nested object
                    obj[key] = {
                        random.choice(keys): random.choice(values),
                        random.choice(keys): random.randint(0, 1000)
                    }
                elif random.random() < 0.5:
                    obj[key] = random.choice(values)
                else:
                    obj[key] = random.randint(0, 10000)

            text = json.dumps(obj, ensure_ascii=False)
            samples.append({'text': text, 'category': 'json'})
        return samples

    def generate_xml(self, count: int = 500) -> List[Dict]:
        """XML 생성"""
        samples = []
        tags = ['user', 'item', 'data', 'record', 'entry', 'node', 'element']
        attrs = ['id', 'name', 'type', 'status', 'value']

        for i in range(count):
            root_tag = random.choice(tags)
            num_children = random.randint(1, 5)

            xml = f"<{root_tag}>"
            for _ in range(num_children):
                child_tag = random.choice(tags)
                if random.random() < 0.5:
                    attr = random.choice(attrs)
                    val = random.randint(1, 999)
                    xml += f"<{child_tag} {attr}=\"{val}\">content</{child_tag}>"
                else:
                    xml += f"<{child_tag}>{random.randint(0, 9999)}</{child_tag}>"
            xml += f"</{root_tag}>"

            samples.append({'text': xml, 'category': 'xml'})
        return samples

    def generate_csv_row(self, count: int = 500) -> List[Dict]:
        """CSV 행 생성"""
        samples = []
        for i in range(count):
            num_cols = random.randint(3, 10)
            values = []
            for _ in range(num_cols):
                if random.random() < 0.5:
                    values.append(str(random.randint(0, 9999)))
                else:
                    values.append(''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))))

            text = ','.join(values)
            samples.append({'text': text, 'category': 'csv_row'})
        return samples

    def generate_korean_sentence(self, count: int = 2000) -> List[Dict]:
        """한글 문장 생성"""
        samples = []
        subjects = ['나는', '그는', '그녀는', '우리는', '그들은', '학생은', '선생님은', '사람들은']
        objects = ['책을', '음식을', '영화를', '음악을', '운동을', '공부를', '일을', '게임을']
        verbs = ['좋아합니다', '싫어합니다', '합니다', '봅니다', '듣습니다', '먹습니다', '합니다']
        adjectives = ['아름다운', '큰', '작은', '좋은', '나쁜', '재미있는', '지루한', '흥미로운']
        nouns = ['날씨', '하루', '시간', '장소', '사람', '이야기', '경험', '생각']

        for i in range(count):
            pattern = random.randint(0, 2)
            if pattern == 0:
                # Subject + Object + Verb
                text = f"{random.choice(subjects)} {random.choice(objects)} {random.choice(verbs)}."
            elif pattern == 1:
                # Adjective + Noun
                text = f"{random.choice(adjectives)} {random.choice(nouns)}입니다."
            else:
                # Complex sentence
                text = f"{random.choice(subjects)} {random.choice(adjectives)} {random.choice(nouns)}에서 {random.choice(objects)} {random.choice(verbs)}."

            samples.append({'text': text, 'category': 'korean_sentence'})
        return samples

    def generate_english_sentence(self, count: int = 2000) -> List[Dict]:
        """영어 문장 생성"""
        samples = []
        subjects = ['I', 'He', 'She', 'We', 'They', 'The student', 'The teacher', 'People']
        verbs = ['like', 'love', 'hate', 'enjoy', 'watch', 'read', 'study', 'play']
        objects = ['books', 'movies', 'music', 'sports', 'games', 'food', 'coffee', 'tea']
        adjectives = ['beautiful', 'big', 'small', 'good', 'bad', 'interesting', 'boring', 'exciting']
        nouns = ['day', 'time', 'place', 'person', 'story', 'experience', 'idea', 'thing']

        for i in range(count):
            pattern = random.randint(0, 2)
            if pattern == 0:
                # Simple sentence
                text = f"{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}."
            elif pattern == 1:
                # With adjective
                text = f"It is a {random.choice(adjectives)} {random.choice(nouns)}."
            else:
                # Complex sentence
                text = f"{random.choice(subjects)} {random.choice(verbs)} the {random.choice(adjectives)} {random.choice(objects)}."

            samples.append({'text': text, 'category': 'english_sentence'})
        return samples

    def generate_chinese_sentence(self, count: int = 500) -> List[Dict]:
        """중국어 문장 생성"""
        samples = []
        subjects = ['我', '他', '她', '我们', '他们', '学生', '老师', '人们']
        verbs = ['喜欢', '讨厌', '看', '听', '吃', '学习', '工作', '玩']
        objects = ['书', '电影', '音乐', '运动', '游戏', '食物', '咖啡', '茶']

        for i in range(count):
            text = f"{random.choice(subjects)}{random.choice(verbs)}{random.choice(objects)}。"
            samples.append({'text': text, 'category': 'chinese_sentence'})
        return samples

    def generate_japanese_sentence(self, count: int = 500) -> List[Dict]:
        """일본어 문장 생성"""
        samples = []
        subjects = ['私は', '彼は', '彼女は', '私たちは', '学生は', '先生は']
        objects = ['本を', '映画を', '音楽を', '運動を', 'ゲームを', '食べ物を']
        verbs = ['好きです', '嫌いです', '見ます', '聞きます', '食べます', '勉強します']

        for i in range(count):
            text = f"{random.choice(subjects)}{random.choice(objects)}{random.choice(verbs)}。"
            samples.append({'text': text, 'category': 'japanese_sentence'})
        return samples

    def generate_mixed_language(self, count: int = 500) -> List[Dict]:
        """다국어 혼합 텍스트 생성"""
        samples = []
        for i in range(count):
            parts = []
            if random.random() < 0.5:
                parts.append(random.choice(['Hello', 'Hi', 'Welcome']))
            if random.random() < 0.5:
                parts.append(random.choice(['안녕하세요', '감사합니다', '환영합니다']))
            if random.random() < 0.5:
                parts.append(random.choice(['你好', '谢谢', '欢迎']))

            text = ' '.join(parts) if parts else 'Hello 안녕하세요'
            samples.append({'text': text, 'category': 'mixed_language'})
        return samples

    def generate_code_javascript(self, count: int = 500) -> List[Dict]:
        """JavaScript 코드 생성"""
        samples = []
        for i in range(count):
            var_name = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 10)))
            value = random.randint(0, 100)

            patterns = [
                f"const {var_name} = {value};",
                f"function {var_name}() {{ return {value}; }}",
                f"let {var_name} = [{', '.join(str(random.randint(0, 100)) for _ in range(3))}];",
                f"if ({var_name} > {value}) {{ console.log('{var_name}'); }}"
            ]

            text = random.choice(patterns)
            samples.append({'text': text, 'category': 'code_javascript'})
        return samples

    def generate_code_python(self, count: int = 500) -> List[Dict]:
        """Python 코드 생성"""
        samples = []
        for i in range(count):
            var_name = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 10)))
            value = random.randint(0, 100)

            patterns = [
                f"{var_name} = {value}",
                f"def {var_name}():\n    return {value}",
                f"{var_name} = [{', '.join(str(random.randint(0, 100)) for _ in range(3))}]",
                f"if {var_name} > {value}:\n    print('{var_name}')"
            ]

            text = random.choice(patterns)
            samples.append({'text': text, 'category': 'code_python'})
        return samples

    def generate_code_sql(self, count: int = 300) -> List[Dict]:
        """SQL 쿼리 생성"""
        samples = []
        tables = ['users', 'products', 'orders', 'items', 'customers', 'sales']
        columns = ['id', 'name', 'email', 'price', 'quantity', 'status', 'created_at']

        for i in range(count):
            table = random.choice(tables)
            col = random.choice(columns)

            patterns = [
                f"SELECT * FROM {table};",
                f"SELECT {col} FROM {table} WHERE id = {random.randint(1, 100)};",
                f"INSERT INTO {table} ({col}) VALUES ({random.randint(1, 100)});",
                f"UPDATE {table} SET {col} = {random.randint(1, 100)} WHERE id = {random.randint(1, 100)};"
            ]

            text = random.choice(patterns)
            samples.append({'text': text, 'category': 'code_sql'})
        return samples

    def generate_hash_string(self, count: int = 500) -> List[Dict]:
        """해시 문자열 생성"""
        samples = []
        for i in range(count):
            hash_type = random.choice(['md5', 'sha256', 'uuid'])

            if hash_type == 'md5':
                text = hashlib.md5(str(random.randint(0, 999999)).encode()).hexdigest()
            elif hash_type == 'sha256':
                text = hashlib.sha256(str(random.randint(0, 999999)).encode()).hexdigest()
            else:
                text = str(uuid.uuid4())

            samples.append({'text': text, 'category': 'hash_string'})
        return samples

    def generate_base64(self, count: int = 300) -> List[Dict]:
        """Base64 문자열 생성"""
        samples = []
        for i in range(count):
            data = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(10, 50)))
            text = base64.b64encode(data.encode()).decode()
            samples.append({'text': text, 'category': 'base64'})
        return samples

    def generate_filepath(self, count: int = 500) -> List[Dict]:
        """파일 경로 생성"""
        samples = []
        dirs = ['home', 'user', 'documents', 'projects', 'data', 'files', 'src', 'lib']
        files = ['file', 'document', 'data', 'config', 'script', 'module']
        extensions = ['txt', 'py', 'js', 'json', 'csv', 'log', 'md', 'yaml']

        for i in range(count):
            if random.random() < 0.5:
                # Unix path
                depth = random.randint(2, 5)
                path_parts = random.sample(dirs, depth)
                filename = f"{random.choice(files)}{random.randint(1, 99)}.{random.choice(extensions)}"
                text = '/' + '/'.join(path_parts) + '/' + filename
            else:
                # Windows path
                drive = random.choice(['C:', 'D:', 'E:'])
                depth = random.randint(2, 4)
                path_parts = random.sample(dirs, depth)
                filename = f"{random.choice(files)}{random.randint(1, 99)}.{random.choice(extensions)}"
                text = drive + '\\' + '\\'.join(path_parts) + '\\' + filename

            samples.append({'text': text, 'category': 'filepath'})
        return samples

    def generate_number_integer(self, count: int = 300) -> List[Dict]:
        """정수 생성"""
        samples = []
        for i in range(count):
            num = random.randint(0, int(1e12))
            samples.append({'text': str(num), 'category': 'number_integer'})
        return samples

    def generate_number_decimal(self, count: int = 300) -> List[Dict]:
        """소수 생성"""
        samples = []
        for i in range(count):
            precision = random.randint(1, 6)
            num = round(random.uniform(0, 10000), precision)
            samples.append({'text': str(num), 'category': 'number_decimal'})
        return samples

    def generate_number_formatted(self, count: int = 300) -> List[Dict]:
        """포맷된 숫자 생성"""
        samples = []
        for i in range(count):
            if random.random() < 0.5:
                # Comma-separated
                num = random.randint(1000, 9999999)
                text = f"{num:,}"
            else:
                # Percentage
                num = round(random.uniform(0, 100), 2)
                text = f"{num}%"

            samples.append({'text': text, 'category': 'number_formatted'})
        return samples

    def generate_single_word(self, count: int = 500) -> List[Dict]:
        """단일 단어 생성"""
        samples = []
        for i in range(count):
            length = random.randint(3, 15)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            samples.append({'text': word, 'category': 'single_word'})
        return samples

    def generate_random_string(self, count: int = 500) -> List[Dict]:
        """랜덤 문자열 생성"""
        samples = []
        for i in range(count):
            length = random.randint(10, 100)
            chars = string.ascii_letters + string.digits + string.punctuation + ' '
            text = ''.join(random.choices(chars, k=length))
            samples.append({'text': text, 'category': 'random_string'})
        return samples

    def generate_all(self) -> List[Dict]:
        """모든 카테고리 데이터 생성"""
        all_samples = []

        print("Generating URL samples...")
        all_samples.extend(self.generate_url(1000))

        print("Generating Email samples...")
        all_samples.extend(self.generate_email(1000))

        print("Generating Phone samples...")
        all_samples.extend(self.generate_phone(500))

        print("Generating Date samples...")
        all_samples.extend(self.generate_date(500))

        print("Generating IPv4 samples...")
        all_samples.extend(self.generate_ipv4(300))

        print("Generating JSON samples...")
        all_samples.extend(self.generate_json(1000))

        print("Generating XML samples...")
        all_samples.extend(self.generate_xml(500))

        print("Generating CSV row samples...")
        all_samples.extend(self.generate_csv_row(500))

        print("Generating Korean sentence samples...")
        all_samples.extend(self.generate_korean_sentence(2000))

        print("Generating English sentence samples...")
        all_samples.extend(self.generate_english_sentence(2000))

        print("Generating Chinese sentence samples...")
        all_samples.extend(self.generate_chinese_sentence(500))

        print("Generating Japanese sentence samples...")
        all_samples.extend(self.generate_japanese_sentence(500))

        print("Generating Mixed language samples...")
        all_samples.extend(self.generate_mixed_language(500))

        print("Generating JavaScript code samples...")
        all_samples.extend(self.generate_code_javascript(500))

        print("Generating Python code samples...")
        all_samples.extend(self.generate_code_python(500))

        print("Generating SQL code samples...")
        all_samples.extend(self.generate_code_sql(300))

        print("Generating Hash string samples...")
        all_samples.extend(self.generate_hash_string(500))

        print("Generating Base64 samples...")
        all_samples.extend(self.generate_base64(300))

        print("Generating Filepath samples...")
        all_samples.extend(self.generate_filepath(500))

        print("Generating Integer samples...")
        all_samples.extend(self.generate_number_integer(300))

        print("Generating Decimal samples...")
        all_samples.extend(self.generate_number_decimal(300))

        print("Generating Formatted number samples...")
        all_samples.extend(self.generate_number_formatted(300))

        print("Generating Single word samples...")
        all_samples.extend(self.generate_single_word(500))

        print("Generating Random string samples...")
        all_samples.extend(self.generate_random_string(500))

        # Shuffle
        random.shuffle(all_samples)

        print(f"\nTotal samples generated: {len(all_samples)}")
        return all_samples
