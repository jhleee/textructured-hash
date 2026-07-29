"""Independent holdout data generator for out-of-distribution validation.

This generator is INTENTIONALLY INDEPENDENT from src/data/generator.py.
It uses different seeds, different template vocabularies, and introduces
4 entirely new category families not present in the development set.

DO NOT import from src/data/generator.py — isolation is critical.
"""

import json
import random
import string
import argparse
import hashlib
import uuid
import base64
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict


class HoldoutGenerator:
    """Independent holdout dataset generator with fresh vocabulary."""

    def __init__(self, seed: int = 2024):
        random.seed(seed)
        self.seed = seed


    # ===== NEW UNSEEN CATEGORY FAMILIES =====

    def generate_markdown(self, count: int = 400) -> list:
        """Markdown-formatted text (headings, lists, bold, links)."""
        samples = []
        headings = ['Introduction', 'Setup', 'Usage', 'API Reference', 'FAQ',
                    'Troubleshooting', 'Contributing', 'License', 'Changelog']
        items = ['Install dependencies', 'Configure settings', 'Run tests',
                 'Deploy to production', 'Check logs', 'Update packages']
        links = ['https://docs.acme.io', 'https://wiki.globex.dev',
                 'https://api.initech.co/v2', 'https://help.umbrella.org']

        for _ in range(count):
            pattern = random.randint(0, 4)
            if pattern == 0:
                level = random.randint(1, 4)
                text = '#' * level + ' ' + random.choice(headings)
            elif pattern == 1:
                n_items = random.randint(2, 5)
                text = '\n'.join(f'- {random.choice(items)}' for _ in range(n_items))
            elif pattern == 2:
                word = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 10)))
                text = f'**{word}** is used for `{word}_config` setup.'
            elif pattern == 3:
                link = random.choice(links)
                label = ''.join(random.choices(string.ascii_lowercase, k=6))
                text = f'See [{label}]({link}) for details.'
            else:
                n_items = random.randint(2, 4)
                text = '\n'.join(f'{i+1}. Step {i+1}: {random.choice(items)}'
                                 for i in range(n_items))
            samples.append({'text': text, 'category': 'markdown'})
        return samples


    def generate_log_entry(self, count: int = 400) -> list:
        """Timestamped log lines (syslog, Apache, JSON-structured)."""
        samples = []
        levels = ['INFO', 'WARN', 'ERROR', 'DEBUG', 'CRITICAL']
        services = ['auth-svc', 'gateway', 'worker-3', 'scheduler', 'db-proxy']
        messages = ['connection established', 'request timeout after 30s',
                    'retry attempt 2/5', 'cache miss for key',
                    'health check passed', 'rate limit exceeded',
                    'transaction committed', 'socket closed unexpectedly']
        ips = [f'192.168.{random.randint(1,254)}.{random.randint(1,254)}'
               for _ in range(20)]

        for _ in range(count):
            ts = datetime(2024, random.randint(1,12), random.randint(1,28),
                         random.randint(0,23), random.randint(0,59),
                         random.randint(0,59))
            pattern = random.randint(0, 2)
            if pattern == 0:
                # syslog-style
                level = random.choice(levels)
                svc = random.choice(services)
                msg = random.choice(messages)
                text = f'{ts.isoformat()} [{level}] {svc}: {msg}'
            elif pattern == 1:
                # Apache-style
                ip = random.choice(ips)
                code = random.choice([200, 301, 404, 500, 503])
                path = '/' + '/'.join(random.choices(['api', 'v1', 'users', 'items'], k=2))
                text = f'{ip} - - [{ts.strftime("%d/%b/%Y:%H:%M:%S")} +0000] "GET {path} HTTP/1.1" {code} {random.randint(100,9999)}'
            else:
                # JSON-structured log
                text = json.dumps({
                    'ts': ts.isoformat(),
                    'level': random.choice(levels).lower(),
                    'service': random.choice(services),
                    'msg': random.choice(messages),
                    'latency_ms': random.randint(1, 5000)
                })
            samples.append({'text': text, 'category': 'log_entry'})
        return samples


    def generate_regex(self, count: int = 300) -> list:
        """Regular expression patterns of varying complexity."""
        samples = []
        char_classes = [r'\d', r'\w', r'\s', r'[a-z]', r'[A-Z]', r'[0-9]',
                        r'[a-zA-Z]', r'[\w.-]', r'[^\s]']
        quantifiers = ['+', '*', '?', '{2,5}', '{1,3}', '{4}']
        anchors = ['^', '$', r'\b']
        groups = [r'(https?://)', r'(\d{1,3}\.){3}', r'([a-z]+)',
                  r'(?:www\.)?', r'(?P<name>\w+)']

        for _ in range(count):
            complexity = random.randint(1, 4)
            parts = []
            if random.random() < 0.4:
                parts.append(random.choice(anchors))
            for _ in range(complexity):
                if random.random() < 0.3:
                    parts.append(random.choice(groups))
                else:
                    cc = random.choice(char_classes)
                    q = random.choice(quantifiers)
                    parts.append(cc + q)
            if random.random() < 0.3:
                parts.append('$')
            text = ''.join(parts)
            samples.append({'text': text, 'category': 'regex'})
        return samples

    def generate_ini_config(self, count: int = 300) -> list:
        """INI/TOML configuration file sections."""
        samples = []
        sections = ['database', 'server', 'logging', 'cache', 'auth',
                    'worker', 'queue', 'metrics', 'storage']
        keys = ['host', 'port', 'timeout', 'max_retries', 'enabled',
                'path', 'level', 'format', 'interval', 'workers']
        values_str = ['localhost', '/var/log/app.log', 'json', 'true', 'false',
                      'production', 'redis://cache:6379', 'DEBUG']
        for _ in range(count):
            section = random.choice(sections)
            n_keys = random.randint(2, 5)
            lines = [f'[{section}]']
            for _ in range(n_keys):
                k = random.choice(keys)
                if random.random() < 0.5:
                    v = str(random.randint(1, 65535))
                else:
                    v = random.choice(values_str)
                lines.append(f'{k} = {v}')
            text = '\n'.join(lines)
            samples.append({'text': text, 'category': 'ini_config'})
        return samples


    # ===== EXISTING CATEGORIES WITH NEW VOCABULARY =====

    def generate_url(self, count: int = 400) -> list:
        """URLs with entirely new domain vocabulary."""
        samples = []
        domains = ['acme', 'globex', 'initech', 'umbrella', 'cyberdyne',
                   'stark', 'waystar', 'hooli', 'piedpiper', 'dunder']
        tlds = ['io', 'cloud', 'app', 'systems', 'tech', 'services', 'run']
        protocols = ['https', 'http']
        paths = ['v2', 'graphql', 'webhook', 'stream', 'health', 'metrics',
                 'rpc', 'ws', 'events', 'admin']

        for _ in range(count):
            proto = random.choice(protocols)
            domain = random.choice(domains) + str(random.randint(1, 99))
            tld = random.choice(tlds)
            if random.random() < 0.7:
                path = '/'.join(random.sample(paths, random.randint(1, 4)))
                url = f'{proto}://{domain}.{tld}/{path}'
                if random.random() < 0.4:
                    url += f'?token={uuid.uuid4().hex[:8]}&limit={random.randint(10,100)}'
            else:
                url = f'{proto}://{domain}.{tld}'
            samples.append({'text': url, 'category': 'url'})
        return samples

    def generate_email(self, count: int = 400) -> list:
        """Emails with new name/domain vocabulary."""
        samples = []
        firsts = ['oliver', 'sophia', 'liam', 'mia', 'noah', 'ava', 'ethan', 'luna']
        lasts = ['chen', 'garcia', 'patel', 'okafor', 'mueller', 'silva', 'tanaka']
        domains = ['acme', 'globex', 'initech', 'proton', 'fastmail', 'tuta']
        tlds = ['io', 'dev', 'tech', 'org', 'co.uk']

        for _ in range(count):
            first = random.choice(firsts)
            last = random.choice(lasts)
            domain = random.choice(domains)
            tld = random.choice(tlds)
            sep = random.choice(['.', '_', '-', '+'])
            if random.random() < 0.4:
                email = f'{first}{sep}{last}{random.randint(1,99)}@{domain}.{tld}'
            else:
                email = f'{first}{sep}{last}@{domain}.{tld}'
            samples.append({'text': email, 'category': 'email'})
        return samples


    def generate_phone(self, count: int = 250) -> list:
        """Phone numbers with shifted format distributions."""
        samples = []
        formats = [
            lambda: f'+1-{random.randint(200,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}',
            lambda: f'+44 {random.randint(1000,9999)} {random.randint(100000,999999)}',
            lambda: f'02-{random.randint(100,999)}-{random.randint(1000,9999)}',
            lambda: f'+81-{random.randint(10,99)}-{random.randint(1000,9999)}-{random.randint(1000,9999)}',
            lambda: f'{random.randint(100,999)}.{random.randint(100,999)}.{random.randint(1000,9999)}',
        ]
        for _ in range(count):
            text = random.choice(formats)()
            samples.append({'text': text, 'category': 'phone'})
        return samples

    def generate_date(self, count: int = 250) -> list:
        """Dates with different base range and new format variants."""
        samples = []
        base = datetime(2022, 6, 1)
        for _ in range(count):
            date = base + timedelta(days=random.randint(0, 900))
            fmt = random.randint(0, 4)
            if fmt == 0:
                text = date.strftime('%Y.%m.%d')
            elif fmt == 1:
                text = date.strftime('%b %d, %Y')
            elif fmt == 2:
                text = date.strftime('%d-%b-%Y')
            elif fmt == 3:
                text = f'{date.year}/{date.month:02d}/{date.day:02d}'
            else:
                text = date.isoformat()
            samples.append({'text': text, 'category': 'date'})
        return samples

    def generate_ipv4(self, count: int = 200) -> list:
        """IPv4 addresses (same structure, new seed)."""
        samples = []
        for _ in range(count):
            ip = f'{random.randint(10,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}'
            samples.append({'text': ip, 'category': 'ipv4'})
        return samples


    def generate_json(self, count: int = 400) -> list:
        """JSON with new key/value vocabulary."""
        samples = []
        keys = ['tenant_id', 'region', 'ttl', 'priority', 'payload',
                'version', 'checksum', 'retries', 'source', 'metadata']
        str_vals = ['pending', 'us-east-1', 'queued', 'v2.1.0',
                    'processing', 'ap-south-1', 'completed']
        for _ in range(count):
            n_keys = random.randint(2, 7)
            obj = {}
            for k in random.sample(keys, min(n_keys, len(keys))):
                if random.random() < 0.3:
                    obj[k] = random.choice(str_vals)
                elif random.random() < 0.5:
                    obj[k] = random.randint(0, 100000)
                else:
                    obj[k] = [random.randint(0, 99) for _ in range(random.randint(2, 4))]
            text = json.dumps(obj, ensure_ascii=False)
            samples.append({'text': text, 'category': 'json'})
        return samples

    def generate_xml(self, count: int = 250) -> list:
        """XML with new tag vocabulary."""
        samples = []
        tags = ['service', 'endpoint', 'config', 'param', 'route', 'handler']
        attrs = ['version', 'timeout', 'enabled', 'priority', 'region']
        for _ in range(count):
            root = random.choice(tags)
            n_children = random.randint(1, 4)
            xml = f'<{root} xmlns="urn:holdout:test">'
            for _ in range(n_children):
                child = random.choice(tags)
                attr = random.choice(attrs)
                val = random.randint(1, 9999)
                xml += f'<{child} {attr}="{val}">{random.choice(["on","off","auto"])}</{child}>'
            xml += f'</{root}>'
            samples.append({'text': xml, 'category': 'xml'})
        return samples

    def generate_csv_row(self, count: int = 250) -> list:
        """CSV rows with different column counts and content."""
        samples = []
        for _ in range(count):
            n_cols = random.randint(4, 12)
            values = []
            for _ in range(n_cols):
                choice = random.random()
                if choice < 0.3:
                    values.append(str(random.randint(-999, 99999)))
                elif choice < 0.6:
                    values.append(f'{random.uniform(-100, 100):.2f}')
                else:
                    values.append(''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 12))))
            text = ','.join(values)
            samples.append({'text': text, 'category': 'csv_row'})
        return samples


    def generate_korean_sentence(self, count: int = 500) -> list:
        """Korean sentences with new template vocabulary."""
        samples = []
        subjects = ['회사는', '팀원은', '시스템이', '프로젝트가', '고객은', '엔지니어가', '매니저는', '서버가']
        objects = ['보고서를', '프레젠테이션을', '배포를', '리뷰를', '미팅을', '업데이트를', '분석을']
        verbs = ['완료했습니다', '시작합니다', '진행 중입니다', '예정입니다', '취소되었습니다', '승인했습니다']
        adverbs = ['빠르게', '성공적으로', '즉시', '신중하게', '효율적으로']

        for _ in range(count):
            p = random.randint(0, 2)
            if p == 0:
                text = f'{random.choice(subjects)} {random.choice(objects)} {random.choice(verbs)}'
            elif p == 1:
                text = f'{random.choice(subjects)} {random.choice(adverbs)} {random.choice(objects)} {random.choice(verbs)}'
            else:
                text = f'{random.choice(adverbs)} {random.choice(subjects)} {random.choice(objects)} {random.choice(verbs)}'
            samples.append({'text': text, 'category': 'korean_sentence'})
        return samples

    def generate_english_sentence(self, count: int = 500) -> list:
        """English sentences with new vocabulary."""
        samples = []
        subjects = ['The system', 'Our team', 'The pipeline', 'A user', 'The server',
                    'The deployment', 'An engineer', 'The scheduler']
        verbs = ['processes', 'handles', 'validates', 'transforms', 'monitors',
                 'schedules', 'deploys', 'optimizes']
        objects = ['incoming requests', 'data streams', 'config files', 'test suites',
                   'service endpoints', 'log entries', 'metric alerts']
        adverbs = ['efficiently', 'reliably', 'automatically', 'concurrently', 'securely']

        for _ in range(count):
            p = random.randint(0, 2)
            if p == 0:
                text = f'{random.choice(subjects)} {random.choice(verbs)} {random.choice(objects)}.'
            elif p == 1:
                text = f'{random.choice(subjects)} {random.choice(adverbs)} {random.choice(verbs)} {random.choice(objects)}.'
            else:
                text = f'{random.choice(adverbs).capitalize()}, {random.choice(subjects).lower()} {random.choice(verbs)} {random.choice(objects)}.'
            samples.append({'text': text, 'category': 'english_sentence'})
        return samples


    def generate_chinese_sentence(self, count: int = 250) -> list:
        """Chinese sentences with new vocabulary."""
        samples = []
        subjects = ['系统', '团队', '服务器', '用户', '管理员', '调度器']
        verbs = ['处理', '验证', '部署', '监控', '优化', '记录']
        objects = ['请求', '数据', '配置', '日志', '指标', '事件']
        for _ in range(count):
            text = f'{random.choice(subjects)}{random.choice(verbs)}了{random.choice(objects)}。'
            samples.append({'text': text, 'category': 'chinese_sentence'})
        return samples

    def generate_japanese_sentence(self, count: int = 250) -> list:
        """Japanese sentences with new vocabulary."""
        samples = []
        subjects = ['システムは', 'チームは', 'サーバーは', 'ユーザーは', '管理者は']
        objects = ['リクエストを', 'データを', '設定を', 'ログを', 'アラートを']
        verbs = ['処理します', '検証します', 'デプロイします', '監視します', '最適化します']
        for _ in range(count):
            text = f'{random.choice(subjects)}{random.choice(objects)}{random.choice(verbs)}。'
            samples.append({'text': text, 'category': 'japanese_sentence'})
        return samples

    def generate_mixed_language(self, count: int = 250) -> list:
        """Mixed-language text with new vocabulary."""
        samples = []
        for _ in range(count):
            parts = []
            if random.random() < 0.5:
                parts.append(random.choice(['Deploy', 'Monitor', 'Check', 'Restart']))
            if random.random() < 0.5:
                parts.append(random.choice(['배포 완료', '점검 필요', '모니터링 중']))
            if random.random() < 0.5:
                parts.append(random.choice(['部署完成', '检查中', '已优化']))
            text = ' '.join(parts) if parts else 'Deploy 배포 完成'
            samples.append({'text': text, 'category': 'mixed_language'})
        return samples


    def generate_code_javascript(self, count: int = 250) -> list:
        """JavaScript code with new patterns."""
        samples = []
        for _ in range(count):
            var = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 9)))
            val = random.randint(0, 200)
            patterns = [
                f'const {var} = async () => {{ return await fetch("/api/{var}"); }};',
                f'export function {var}(req, res) {{ res.json({{ ok: true }}); }}',
                f'const {var} = [{", ".join(str(random.randint(0,99)) for _ in range(4))}];',
                f'try {{ {var}({val}); }} catch (err) {{ console.error(err); }}',
                f'import {{ {var} }} from "@scope/{var}-lib";',
            ]
            text = random.choice(patterns)
            samples.append({'text': text, 'category': 'code_javascript'})
        return samples

    def generate_code_python(self, count: int = 250) -> list:
        """Python code with new patterns."""
        samples = []
        for _ in range(count):
            var = ''.join(random.choices(string.ascii_lowercase, k=random.randint(4, 9)))
            val = random.randint(0, 200)
            patterns = [
                f'async def {var}(session: aiohttp.ClientSession) -> dict:\n    return await session.get("/")',
                f'class {var.capitalize()}Error(Exception):\n    pass',
                f'{var}: list[int] = [{", ".join(str(random.randint(0,99)) for _ in range(4))}]',
                f'with open("{var}.log", "a") as f:\n    f.write(str({val}))',
                f'from pathlib import Path\n{var}_path = Path("/opt/{var}")',
            ]
            text = random.choice(patterns)
            samples.append({'text': text, 'category': 'code_python'})
        return samples

    def generate_code_sql(self, count: int = 200) -> list:
        """SQL with new table/column vocabulary."""
        samples = []
        tables = ['deployments', 'incidents', 'metrics', 'tenants', 'audit_log', 'pipelines']
        columns = ['tenant_id', 'severity', 'latency_p99', 'region', 'created_at', 'pipeline_run']
        for _ in range(count):
            table = random.choice(tables)
            col = random.choice(columns)
            patterns = [
                f'SELECT {col}, COUNT(*) FROM {table} GROUP BY {col} HAVING COUNT(*) > {random.randint(1,100)};',
                f'DELETE FROM {table} WHERE {col} < {random.randint(1,1000)} AND region = \'us-west-2\';',
                f'CREATE INDEX idx_{table}_{col} ON {table}({col});',
                f'ALTER TABLE {table} ADD COLUMN {col}_v2 JSONB DEFAULT \'{{}}\';',
            ]
            text = random.choice(patterns)
            samples.append({'text': text, 'category': 'code_sql'})
        return samples


    def generate_hash_string(self, count: int = 250) -> list:
        """Hash strings (same structure, new seed)."""
        samples = []
        for _ in range(count):
            src = str(random.randint(1000000, 9999999))
            ht = random.choice(['md5', 'sha256', 'sha1', 'uuid'])
            if ht == 'md5':
                text = hashlib.md5(src.encode()).hexdigest()
            elif ht == 'sha256':
                text = hashlib.sha256(src.encode()).hexdigest()
            elif ht == 'sha1':
                text = hashlib.sha1(src.encode()).hexdigest()
            else:
                text = str(uuid.uuid4())
            samples.append({'text': text, 'category': 'hash_string'})
        return samples

    def generate_base64(self, count: int = 200) -> list:
        """Base64 strings (new seed)."""
        samples = []
        for _ in range(count):
            length = random.randint(12, 64)
            raw = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
            text = base64.b64encode(raw.encode()).decode()
            samples.append({'text': text, 'category': 'base64'})
        return samples

    def generate_filepath(self, count: int = 250) -> list:
        """File paths with new directory/file vocabulary."""
        samples = []
        dirs = ['opt', 'srv', 'workspace', 'deploy', 'artifacts', 'cache', 'tmp', 'volumes']
        files = ['service', 'worker', 'handler', 'pipeline', 'index', 'main']
        extensions = ['go', 'rs', 'ts', 'toml', 'lock', 'env', 'dockerfile', 'tf']
        for _ in range(count):
            if random.random() < 0.6:
                depth = random.randint(2, 5)
                parts = random.choices(dirs, k=depth)
                fname = f'{random.choice(files)}.{random.choice(extensions)}'
                text = '/' + '/'.join(parts) + '/' + fname
            else:
                drive = random.choice(['C:', 'D:'])
                depth = random.randint(2, 4)
                parts = random.choices(dirs, k=depth)
                fname = f'{random.choice(files)}.{random.choice(extensions)}'
                text = drive + '\\' + '\\'.join(parts) + '\\' + fname
            samples.append({'text': text, 'category': 'filepath'})
        return samples


    def generate_number_integer(self, count: int = 200) -> list:
        """Integer numbers (new range)."""
        samples = []
        for _ in range(count):
            n = random.randint(-10**9, 10**12)
            samples.append({'text': str(n), 'category': 'number_integer'})
        return samples

    def generate_number_decimal(self, count: int = 200) -> list:
        """Decimal numbers (new precision range)."""
        samples = []
        for _ in range(count):
            precision = random.randint(1, 8)
            n = round(random.uniform(-5000, 50000), precision)
            samples.append({'text': str(n), 'category': 'number_decimal'})
        return samples

    def generate_number_formatted(self, count: int = 200) -> list:
        """Formatted numbers (new patterns)."""
        samples = []
        for _ in range(count):
            choice = random.random()
            if choice < 0.3:
                n = random.randint(10000, 99999999)
                text = f'{n:,}'
            elif choice < 0.6:
                text = f'{random.uniform(0, 100):.1f}%'
            else:
                text = f'${random.randint(1, 9999)}.{random.randint(0,99):02d}'
            samples.append({'text': text, 'category': 'number_formatted'})
        return samples

    def generate_single_word(self, count: int = 250) -> list:
        """Single words (new length distribution)."""
        samples = []
        for _ in range(count):
            length = random.randint(2, 18)
            word = ''.join(random.choices(string.ascii_lowercase, k=length))
            samples.append({'text': word, 'category': 'single_word'})
        return samples

    def generate_random_string(self, count: int = 250) -> list:
        """Random strings (wider char set)."""
        samples = []
        for _ in range(count):
            length = random.randint(8, 120)
            chars = string.ascii_letters + string.digits + string.punctuation + ' \t'
            text = ''.join(random.choices(chars, k=length))
            samples.append({'text': text, 'category': 'random_string'})
        return samples


    # ===== MAIN GENERATION =====

    def generate_all(self) -> list:
        """Generate all holdout samples (24 existing + 4 new families)."""
        all_samples = []

        # Existing 24 categories (new vocabulary)
        print('Generating holdout: existing categories (new vocabulary)...')
        all_samples.extend(self.generate_url(400))
        all_samples.extend(self.generate_email(400))
        all_samples.extend(self.generate_phone(250))
        all_samples.extend(self.generate_date(250))
        all_samples.extend(self.generate_ipv4(200))
        all_samples.extend(self.generate_json(400))
        all_samples.extend(self.generate_xml(250))
        all_samples.extend(self.generate_csv_row(250))
        all_samples.extend(self.generate_korean_sentence(500))
        all_samples.extend(self.generate_english_sentence(500))
        all_samples.extend(self.generate_chinese_sentence(250))
        all_samples.extend(self.generate_japanese_sentence(250))
        all_samples.extend(self.generate_mixed_language(250))
        all_samples.extend(self.generate_code_javascript(250))
        all_samples.extend(self.generate_code_python(250))
        all_samples.extend(self.generate_code_sql(200))
        all_samples.extend(self.generate_hash_string(250))
        all_samples.extend(self.generate_base64(200))
        all_samples.extend(self.generate_filepath(250))
        all_samples.extend(self.generate_number_integer(200))
        all_samples.extend(self.generate_number_decimal(200))
        all_samples.extend(self.generate_number_formatted(200))
        all_samples.extend(self.generate_single_word(250))
        all_samples.extend(self.generate_random_string(250))

        # 4 NEW unseen families
        print('Generating holdout: new unseen category families...')
        all_samples.extend(self.generate_markdown(400))
        all_samples.extend(self.generate_log_entry(400))
        all_samples.extend(self.generate_regex(300))
        all_samples.extend(self.generate_ini_config(300))

        random.shuffle(all_samples)
        print(f'Total holdout samples: {len(all_samples)}')
        return all_samples



def generate_holdout_pairs(samples, n_positive=3000, n_negative=3000, seed=7777):
    """Generate positive/negative pairs from holdout samples.

    Independent pair generation with its own seed.
    """
    random.seed(seed)

    by_category = defaultdict(list)
    for s in samples:
        by_category[s['category']].append(s)

    categories = sorted(by_category.keys())
    print(f'Holdout categories: {len(categories)}')
    for cat in categories:
        print(f'  {cat}: {len(by_category[cat])} samples')

    pairs = []

    # Positive pairs (same category)
    pairs_per_cat = max(1, n_positive // len(categories))
    for cat in categories:
        cat_samples = by_category[cat]
        if len(cat_samples) < 2:
            continue
        n_cat_pairs = min(pairs_per_cat, len(cat_samples) * (len(cat_samples) - 1) // 2)
        for _ in range(n_cat_pairs):
            s1, s2 = random.sample(cat_samples, 2)
            pairs.append({
                'text1': s1['text'],
                'text2': s2['text'],
                'category1': cat,
                'category2': cat,
                'label': 1.0,
                'pair_type': 'intra_category'
            })

    # Trim to exact count
    if len(pairs) > n_positive:
        pairs = random.sample(pairs, n_positive)

    # Negative pairs (different categories)
    neg_pairs = []
    for _ in range(n_negative):
        cat1, cat2 = random.sample(categories, 2)
        s1 = random.choice(by_category[cat1])
        s2 = random.choice(by_category[cat2])
        neg_pairs.append({
            'text1': s1['text'],
            'text2': s2['text'],
            'category1': cat1,
            'category2': cat2,
            'label': 0.0,
            'pair_type': 'inter_category'
        })

    pairs.extend(neg_pairs)
    random.shuffle(pairs)

    n_pos = sum(1 for p in pairs if p['label'] == 1.0)
    n_neg = sum(1 for p in pairs if p['label'] == 0.0)
    print(f'\nHoldout pairs: {len(pairs)} (pos={n_pos}, neg={n_neg})')
    return pairs


def main():
    parser = argparse.ArgumentParser(
        description='Generate independent holdout dataset for OOD validation')
    parser.add_argument('--output', type=str, default='data/holdout',
                        help='Output directory')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Generator seed (must differ from dev seed=42)')
    parser.add_argument('--pair_seed', type=int, default=7777,
                        help='Pair generation seed')
    parser.add_argument('--n_positive', type=int, default=3000,
                        help='Number of positive pairs')
    parser.add_argument('--n_negative', type=int, default=3000,
                        help='Number of negative pairs')
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 60)
    print('Independent Holdout Data Generation')
    print('=' * 60)
    print(f'  Generator seed: {args.seed}')
    print(f'  Pair seed:      {args.pair_seed}')
    print(f'  Output:         {args.output}')
    print()

    # Generate samples
    gen = HoldoutGenerator(seed=args.seed)
    samples = gen.generate_all()

    # Save raw samples
    samples_path = output_dir / 'samples.jsonl'
    with open(samples_path, 'w', encoding='utf-8') as f:
        for s in samples:
            f.write(json.dumps(s, ensure_ascii=False) + '\n')
    print(f'\nSamples saved: {samples_path}')

    # Generate pairs
    pairs = generate_holdout_pairs(samples, args.n_positive, args.n_negative, args.pair_seed)

    # Save pairs
    pairs_path = output_dir / 'pairs.jsonl'
    with open(pairs_path, 'w', encoding='utf-8') as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + '\n')
    print(f'Pairs saved: {pairs_path}')

    # Save metadata
    meta = {
        'generator_seed': args.seed,
        'pair_seed': args.pair_seed,
        'n_samples': len(samples),
        'n_pairs': len(pairs),
        'n_positive': sum(1 for p in pairs if p['label'] == 1.0),
        'n_negative': sum(1 for p in pairs if p['label'] == 0.0),
        'categories': sorted(set(s['category'] for s in samples)),
        'new_families': ['markdown', 'log_entry', 'regex', 'ini_config'],
        'existing_families': [c for c in sorted(set(s['category'] for s in samples))
                              if c not in ['markdown', 'log_entry', 'regex', 'ini_config']],
    }
    meta_path = output_dir / 'metadata.json'
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    print(f'Metadata saved: {meta_path}')

    print('\n' + '=' * 60)
    print('Holdout generation complete.')
    print('=' * 60)


if __name__ == '__main__':
    main()
