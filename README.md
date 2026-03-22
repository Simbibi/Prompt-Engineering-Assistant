# RAG Assistant

Быстрый старт
----------

1. Скопируйте пример файла окружения и заполните его реальными секретными данными:

```bash
cp .env.example .env
# edit .env and fill API_KEY or LLM_VENDER_SECRET, DATABASE_URL, etc.
```

2. Создайте и активируйте виртуальную среду, установите зависимости:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Запустите приложение или скрипты по мере необходимости (пример):

```bash
uvicorn main:app --reload
# or use your preferred entrypoint
```
