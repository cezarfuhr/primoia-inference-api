# Usar uma imagem base Python mínima
FROM python:3.9-slim

# Definir o diretório de trabalho
WORKDIR /app

# Copiar o arquivo requirements.txt
COPY requirements.txt .

# Instalar as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Copiar o código da aplicação
COPY app.py .

# Expor a porta
EXPOSE 8000

# Comando para rodar a aplicação
CMD ["python", "app.py"]