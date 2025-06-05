import json
import requests
import pandas as pd

from pathlib import Path
from docling.document_converter import DocumentConverter
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode, TesseractCliOcrOptions
from docling.datamodel.base_models import InputFormat
from pdfalign import align

def extract_pdf_content(pdf_path):
    """
    Extrai o conteúdo do PDF usando Docling
    """
    try:
        # Verificar se o arquivo existe
        if not Path(pdf_path).exists():
            raise FileNotFoundError(f"Arquivo não encontrado: {pdf_path}")
       
        
        # Inicializar o conversor de documentos
        converter = DocumentConverter()
        
        # Converter o PDF
        result = converter.convert(pdf_path)

        forced_column_names = [ "Q", "Negociação", "C/V", "Tipo mercado", "Prazo", "Especificação do título", "Obs. (*)", "Quantidade", "Preço/Ajuste", "Valor/Ajuste", "D/C"]

        tables = result.document.tables
        print(f" num tabelas: {len(tables)}")
        

        for table in tables:
            data = table.export_to_dataframe()            
        # Pode ser necessário remover o header se o Docling detectou errado
            #if any(x in data[0] for x in forced_column_names):
            data = data[1:]

            df = pd.DataFrame(data)
            # Ajuste possíveis desalinhamentos
            for _ in range(len(forced_column_names) - len(df.columns)):
                df[len(df.columns)] = None
            df = df.iloc[:, :len(forced_column_names)]
            df.columns = forced_column_names

            print(df)
        
        # Extrair o texto
        text = result.document.export_to_text()
        
        print("✓ Texto extraído do PDF com sucesso")
        
        # Tentar extrair estrutura JSON
        try:
            # Converter para dicionário
            doc_dict = result.document.export_to_dict()
            
            # Converter para JSON formatado
            pdf_json = json.dumps(doc_dict, ensure_ascii=False, indent=2)
            save_result(pdf_json, "pdf_json")
            
            # Salvar estrutura JSON
            with open("pdf_structure.json", "w", encoding="utf-8") as f:
                f.write(pdf_json)
            
            print("✓ Estrutura JSON extraída e salva em 'pdf_structure.json'")
            return text, pdf_json, True
            
        except Exception as e:
            print(f"⚠ Erro ao extrair estrutura JSON: {e}")
            return text, None, False
            
    except Exception as e:
        print(f"❌ Erro ao processar PDF: {e}")
        return None, None, False

def send_to_llm(prompt, model="llama3.1:8b", url="http://localhost:11434/api/generate"):
    """
    Envia prompt para o modelo LLM local
    """
    try:
        print("🔄 Enviando dados para o modelo LLM...")
        
        response = requests.post(
            url,
            json={
                "model": model,
                "prompt": prompt,
                "stream": False  # Mudado para False para resposta completa
            },
            timeout=300  # Timeout de 5 minutos
        )
        
        response.raise_for_status()  # Levanta exceção para códigos de erro HTTP
        
        # Para stream=False, a resposta vem completa
        data = response.json()
        result = data.get("response", "")
        
        print("✓ Resposta recebida do modelo LLM")
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Erro na comunicação com o LLM: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"❌ Erro ao decodificar resposta JSON: {e}")
        return None

def save_result(result, filename_base="negocios_realizados"):
    """
    Salva o resultado em arquivo JSON ou TXT
    """
    try:
        # Tenta converter para JSON
        parsed_result = json.loads(result)
        
        # Salva como JSON
        json_filename = f"{filename_base}.json"
        with open(json_filename, "w", encoding="utf-8") as f:
            json.dump(parsed_result, f, ensure_ascii=False, indent=2)
        
        print(f"✓ Resultado salvo em '{json_filename}'")
        return True
        
    except json.JSONDecodeError:
        # Se não for JSON válido, salva como texto
        txt_filename = f"{filename_base}.txt"
        with open(txt_filename, "w", encoding="utf-8") as f:
            f.write(result)
        
        print(f"⚠ Resultado não é JSON válido. Salvo como texto em '{txt_filename}'")
        return False

def create_llm_prompt():
    """
    Cria o prompt otimizado para o LLM
    """
    return """Analíse o texto de uma nota de negociação da B3 e estraia os dados que foram negociados. leve em consideração que:

a tabela precisar conter as seguintes 11 colunas: Q, Negociação, C/V, Tipo mercado, Prazo, Especificação do título, Obs. (*), Quantidade, Preço/Ajuste, Valor/Ajuste, D/C.
analise o texto informado e corrija as informações de acordo com essas colunas.
```"""

def main(): 
    """ Função principal """
    print("🚀 Iniciando extração de dados da nota de corretagem...\n")

    # Configurações
    pdf_path = "XPINC_NOTA_NEGOCIACAO_B3_1_2024.pdf"

    # 1. Extrair conteúdo do PDF
    print("📄 Extraindo conteúdo do PDF...")
    print("📄 TESTE...")
    text, pdf_json, has_structure = extract_pdf_content(pdf_path)

    if not text:
        print("❌ Falha ao extrair conteúdo do PDF. Encerrando.")
        return

    print(f"📝 Texto extraído (primeiros 500 caracteres):")
    print(text[:500] + "..." if len(text) > 500 else text)
    print()

    # 2. Criar prompt
    base_prompt = create_llm_prompt()

    if has_structure and pdf_json:
        final_prompt = f"{base_prompt}\n\n o texto da Nota de negociação é:\n{text}"
    else:
        final_prompt = f"{base_prompt}\n\nTexto extraído do PDF:\n{text}"

    # 3. Enviar para LLM
    print("🤖 Processando com modelo LLM...")
    result = send_to_llm(final_prompt)
    #result = None

    if not result:
        print("❌ Falha ao obter resposta do modelo LLM. Encerrando.")
        return

    print("📋 Resultado da extração:")
    print(result)
    print("=============================================================")
    print(text)

    # 4. Salvar resultado
    print("💾 Salvando resultado...")
    save_result(result)

    print("\n✅ Processo concluído com sucesso!")

if __name__ == "__main__":
    main()