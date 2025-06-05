import fitz  # PyMuPDF
import cv2
import numpy as np
import pandas as pd
import yaml
import os
from typing import Dict, List, Tuple, Optional, Any, Union, Set
from dataclasses import dataclass
import camelot


@dataclass
class TableConfig:
    """Representa a configuração de uma tabela para extração."""
    name: str
    header_phrases: List[str]
    header_y_filter: Optional[Tuple[float, float]] = None
    y_jump_threshold: float = 15.0
    page_number: int = 0
    # Novos parâmetros para melhorar a detecção de cabeçalhos
    max_header_y_variance: float = 5.0  # Variação máxima de Y entre cabeçalhos da mesma linha
    min_header_match_ratio: float = 0.7  # Proporção mínima de cabeçalhos que devem ser encontrados
    header_context_words: Optional[List[str]] = None  # Palavras de contexto que devem estar próximas aos cabeçalhos


@dataclass
class TableBoundary:
    """Representa os limites de uma tabela."""
    x1: float
    y1: float
    x2: float
    y2: float
    
    def as_tuple(self) -> Tuple[float, float, float, float]:
        """Retorna os limites como uma tupla."""
        return (self.x1, self.y1, self.x2, self.y2)


class ConfigLoader:
    """Classe responsável por carregar configurações de tabelas a partir de arquivos YAML."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, TableConfig]:
        """
        Carrega configurações de tabelas a partir de um arquivo YAML.
        
        Args:
            config_path: Caminho para o arquivo de configuração YAML
            
        Returns:
            Dicionário de configurações de tabela, indexado por nome
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Arquivo de configuração não encontrado: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as file:
            config_data = yaml.safe_load(file)
        
        table_configs = {}
        
        for table_name, table_config in config_data.get('tables', {}).items():
            header_phrases = table_config.get('header_phrases', [])
            
            header_y_filter = None
            if 'header_y_filter' in table_config:
                y_min = table_config['header_y_filter'].get('min')
                y_max = table_config['header_y_filter'].get('max')
                if y_min is not None and y_max is not None:
                    header_y_filter = (float(y_min), float(y_max))
            
            y_jump_threshold = float(table_config.get('y_jump_threshold', 15.0))
            page_number = int(table_config.get('page_number', 0))
            
            # Novos parâmetros para melhorar a detecção de cabeçalhos
            max_header_y_variance = float(table_config.get('max_header_y_variance', 5.0))
            min_header_match_ratio = float(table_config.get('min_header_match_ratio', 0.7))
            header_context_words = table_config.get('header_context_words')
            
            table_configs[table_name] = TableConfig(
                name=table_name,
                header_phrases=header_phrases,
                header_y_filter=header_y_filter,
                y_jump_threshold=y_jump_threshold,
                page_number=page_number,
                max_header_y_variance=max_header_y_variance,
                min_header_match_ratio=min_header_match_ratio,
                header_context_words=header_context_words
            )
        
        return table_configs


class PDFDocumentHandler:
    """Classe responsável por gerenciar a abertura e fechamento de documentos PDF."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.document = None
        
    def __enter__(self):
        self.document = fitz.open(self.pdf_path)
        return self.document
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.document:
            self.document.close()


class HeaderExtractor:
    """Classe responsável por extrair informações de cabeçalhos de tabelas em PDFs."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
    
    def _find_phrase_in_text(self, words: List[Any], phrase: str, 
                            start_idx: int, y_range_filter: Optional[Tuple[float, float]] = None) -> Optional[List[Any]]:
        """
        Procura uma frase específica a partir de um índice inicial na lista de palavras.
        
        Args:
            words: Lista de palavras do PDF
            phrase: Frase a ser procurada
            start_idx: Índice inicial para começar a busca
            y_range_filter: Opcional, limita a busca a um intervalo vertical (y0, y1)
            
        Returns:
            Lista de informações das palavras que compõem a frase encontrada, ou None se não encontrada
        """
        phrase_parts = phrase.split()
        if start_idx >= len(words) or words[start_idx][4] != phrase_parts[0]:
            return None
        
        word_info = words[start_idx]
        word_bbox = word_info[:4]  # (x0, y0, x1, y1)
        
        if y_range_filter and not (y_range_filter[0] <= word_bbox[1] <= y_range_filter[1]):
            return None
        
        current_phrase_words_info = [word_info]
        
        for j in range(1, len(phrase_parts)):
            if start_idx + j < len(words):
                next_word_info = words[start_idx + j]
                next_word_text = next_word_info[4]
                
                # Verifica proximidade horizontal e vertical
                if (abs(next_word_info[1] - word_bbox[1]) < 5 and 
                    next_word_text == phrase_parts[j]):
                    current_phrase_words_info.append(next_word_info)
                else:
                    return None  # Sequência quebrada
            else:
                return None  # Não há mais palavras
        
        return current_phrase_words_info
    
    def _calculate_bbox_for_phrase(self, phrase_words_info: List[Any]) -> Tuple[float, float, float, float]:
        """
        Calcula a bounding box combinada para uma frase.
        
        Args:
            phrase_words_info: Lista de informações das palavras que compõem a frase
            
        Returns:
            Bounding box (x0, y0, x1, y1) da frase
        """
        min_x0 = min(w[0] for w in phrase_words_info)
        min_y0 = min(w[1] for w in phrase_words_info)
        max_x1 = max(w[2] for w in phrase_words_info)
        max_y1 = max(w[3] for w in phrase_words_info)
        
        return (min_x0, min_y0, max_x1, max_y1)
    
    def _are_headers_in_same_line(self, headers_coords: Dict[str, Tuple[float, float, float, float]], 
                                max_y_variance: float) -> bool:
        """
        Verifica se os cabeçalhos encontrados estão aproximadamente na mesma linha horizontal.
        
        Args:
            headers_coords: Dicionário com as coordenadas dos cabeçalhos
            max_y_variance: Variação máxima permitida no eixo Y
            
        Returns:
            True se os cabeçalhos estiverem aproximadamente na mesma linha
        """
        if not headers_coords:
            return False
        
        y_values = [bbox[1] for bbox in headers_coords.values()]  # y0 de cada cabeçalho
        min_y = min(y_values)
        max_y = max(y_values)
        
        return max_y - min_y <= max_y_variance
    
    def _has_context_words_nearby(self, words: List[Any], headers_coords: Dict[str, Tuple[float, float, float, float]], 
                                context_words: List[str], proximity_threshold: float = 50.0) -> bool:
        """
        Verifica se há palavras de contexto próximas aos cabeçalhos encontrados.
        
        Args:
            words: Lista de palavras do PDF
            headers_coords: Dicionário com as coordenadas dos cabeçalhos
            context_words: Lista de palavras de contexto a serem procuradas
            proximity_threshold: Distância máxima para considerar uma palavra próxima
            
        Returns:
            True se pelo menos uma palavra de contexto for encontrada próxima aos cabeçalhos
        """
        if not context_words or not headers_coords:
            return True  # Se não houver palavras de contexto definidas, considera verdadeiro
        
        # Calcula o centro da área dos cabeçalhos
        all_x = []
        all_y = []
        for bbox in headers_coords.values():
            all_x.extend([bbox[0], bbox[2]])  # x0 e x1
            all_y.extend([bbox[1], bbox[3]])  # y0 e y1
        
        center_x = sum(all_x) / len(all_x)
        center_y = sum(all_y) / len(all_y)
        
        # Procura palavras de contexto próximas
        for word_info in words:
            word_text = word_info[4].lower()
            if any(context_word.lower() in word_text for context_word in context_words):
                word_x = (word_info[0] + word_info[2]) / 2  # Centro X da palavra
                word_y = (word_info[1] + word_info[3]) / 2  # Centro Y da palavra
                
                distance = np.sqrt((word_x - center_x)**2 + (word_y - center_y)**2)
                if distance <= proximity_threshold:
                    return True
        
        return False
    
    def extract_headers(self, page_number: int, header_phrases: List[str], 
                       y_range_filter: Optional[Tuple[float, float]] = None,
                       max_header_y_variance: float = 5.0,
                       min_header_match_ratio: float = 0.7,
                       header_context_words: Optional[List[str]] = None) -> Dict[str, Tuple[float, float, float, float]]:
        """
        Extrai informações detalhadas dos cabeçalhos, incluindo suas coordenadas.
        
        Args:
            page_number: Número da página (baseado em 0)
            header_phrases: Lista de frases de cabeçalho a serem procuradas
            y_range_filter: Opcional, limita a busca a um intervalo vertical (y0, y1)
            max_header_y_variance: Variação máxima permitida no eixo Y entre cabeçalhos
            min_header_match_ratio: Proporção mínima de cabeçalhos que devem ser encontrados
            header_context_words: Palavras de contexto que devem estar próximas aos cabeçalhos
            
        Returns:
            Dicionário com frases de cabeçalho e suas coordenadas (x0, y0, x1, y1)
        """
        # Armazena todas as ocorrências de cabeçalhos encontradas
        all_header_occurrences = []  # Lista de dicionários {phrase: (x0, y0, x1, y1)}
        
        with PDFDocumentHandler(self.pdf_path) as document:
            page = document.load_page(page_number)
            words = page.get_text("words")
            
            # Para cada palavra no documento, verifica se ela inicia algum dos cabeçalhos
            for i, word_info in enumerate(words):
                word_text = word_info[4]
                
                for phrase in header_phrases:
                    phrase_parts = phrase.split()
                    if word_text == phrase_parts[0]:
                        # Tenta encontrar a frase completa
                        phrase_words_info = self._find_phrase_in_text(words, phrase, i, y_range_filter)
                        
                        if phrase_words_info:
                            # Calcula a bounding box para a frase
                            bbox = self._calculate_bbox_for_phrase(phrase_words_info)
                            
                            # Cria um novo conjunto de cabeçalhos para esta ocorrência
                            current_headers = {phrase: bbox}
                            
                            # Procura outros cabeçalhos próximos horizontalmente a este
                            for other_phrase in header_phrases:
                                if other_phrase != phrase:
                                    # Procura na vizinhança (próximas 100 palavras)
                                    for j in range(i+len(phrase_parts), min(i+len(phrase_parts)+100, len(words))):
                                        if words[j][4] == other_phrase.split()[0]:
                                            other_phrase_words_info = self._find_phrase_in_text(
                                                words, other_phrase, j, y_range_filter)
                                            
                                            if other_phrase_words_info:
                                                other_bbox = self._calculate_bbox_for_phrase(other_phrase_words_info)
                                                
                                                # Verifica se está aproximadamente na mesma linha horizontal
                                                if abs(other_bbox[1] - bbox[1]) <= max_header_y_variance:
                                                    current_headers[other_phrase] = other_bbox
                                                    break
                            
                            # Adiciona este conjunto de cabeçalhos às ocorrências encontradas
                            all_header_occurrences.append(current_headers)
        
        # Se não encontrou nenhuma ocorrência, retorna dicionário vazio
        if not all_header_occurrences:
            return {}
        
        # Avalia cada conjunto de cabeçalhos encontrado
        best_headers = None
        best_score = 0
        
        for headers in all_header_occurrences:
            # Calcula a pontuação baseada em vários critérios
            # 1. Número de cabeçalhos encontrados
            match_ratio = len(headers) / len(header_phrases)
            
            # 2. Verifica se estão na mesma linha horizontal
            same_line = self._are_headers_in_same_line(headers, max_header_y_variance)
            
            # 3. Verifica se há palavras de contexto próximas
            has_context = self._has_context_words_nearby(words, headers, header_context_words or [])
            
            # Calcula a pontuação final
            score = match_ratio * (2 if same_line else 1) * (1.5 if has_context else 1)
            
            # Atualiza o melhor conjunto se a pontuação for maior
            if score > best_score and match_ratio >= min_header_match_ratio:
                best_score = score
                best_headers = headers
        
        return best_headers or {}


class TableAreaInferer:
    """Classe responsável por inferir a área da tabela e as coordenadas das colunas."""
    
    def __init__(self, pdf_path: str, header_extractor: HeaderExtractor):
        self.pdf_path = pdf_path
        self.header_extractor = header_extractor
    
    def infer_table_area(self, page_number: int, header_phrases: List[str], 
                         header_y_filter_range: Optional[Tuple[float, float]] = None,
                         y_jump_threshold: float = 15,
                         max_header_y_variance: float = 5.0,
                         min_header_match_ratio: float = 0.7,
                         header_context_words: Optional[List[str]] = None) -> Tuple[
                             Optional[TableBoundary], 
                             Optional[List[float]], 
                             Optional[List[Tuple[str, Tuple[float, float, float, float]]]], 
                             Optional[List[Any]], 
                             Optional[Dict[str, Tuple[float, float, float, float]]]
                         ]:
        """
        Infere a área da tabela e as coordenadas das colunas automaticamente.
        
        Args:
            page_number: Número da página (baseado em 0)
            header_phrases: Lista de frases de cabeçalho a serem procuradas
            header_y_filter_range: Opcional, limita a busca a um intervalo vertical (y0, y1)
            y_jump_threshold: Limiar para detectar saltos verticais entre linhas
            max_header_y_variance: Variação máxima permitida no eixo Y entre cabeçalhos
            min_header_match_ratio: Proporção mínima de cabeçalhos que devem ser encontrados
            header_context_words: Palavras de contexto que devem estar próximas aos cabeçalhos
            
        Returns:
            Tupla contendo:
            - Limites da tabela (TableBoundary)
            - Lista de coordenadas X dos separadores de coluna
            - Lista de cabeçalhos ordenados por X
            - Lista de todas as palavras na página
            - Dicionário com as coordenadas dos cabeçalhos
        """
        all_words_on_page = None
        
        with PDFDocumentHandler(self.pdf_path) as document:
            page = document.load_page(page_number)
            all_words_on_page = page.get_text("words")
        
        headers_coords = self.header_extractor.extract_headers(
            page_number, 
            header_phrases, 
            header_y_filter_range,
            max_header_y_variance,
            min_header_match_ratio,
            header_context_words
        )

        if not headers_coords:
            print("Erro: Não foi possível encontrar os cabeçalhos da tabela.")
            return None, None, None, None, None

        sorted_headers_by_x = sorted(headers_coords.items(), key=lambda item: item[1][0])
        
        # Se não encontrou pelo menos dois cabeçalhos, não é possível inferir a área da tabela
        if len(sorted_headers_by_x) < 2:
            print("Erro: Número insuficiente de cabeçalhos encontrados para inferir a área da tabela.")
            return None, None, None, None, None
        
        first_header_bbox = sorted_headers_by_x[0][1]
        last_header_bbox = sorted_headers_by_x[-1][1]

        # Calcula os limites da tabela com base nos cabeçalhos encontrados
        # Adiciona uma margem à esquerda e à direita
        x1_table = first_header_bbox[0] - 10
        y1_table = min(header[1][1] for header in sorted_headers_by_x)  # Menor y0 entre os cabeçalhos
        x2_table = last_header_bbox[2] + 20
        
        # Calcula a altura média dos cabeçalhos para usar como referência
        header_heights = [header[1][3] - header[1][1] for header in sorted_headers_by_x]
        avg_header_height = sum(header_heights) / len(header_heights)

        column_separators_x_coords = []
        for i in range(len(sorted_headers_by_x) - 1):
            # Usa o ponto médio entre o final de um cabeçalho e o início do próximo
            end_of_current = sorted_headers_by_x[i][1][2]  # x1 do cabeçalho atual
            start_of_next = sorted_headers_by_x[i+1][1][0]  # x0 do próximo cabeçalho
            separator_x = (end_of_current + start_of_next) / 2
            column_separators_x_coords.append(separator_x)
        
        # Encontra o final da tabela analisando saltos verticais
        words_below_headers = None
        with PDFDocumentHandler(self.pdf_path) as document:
            page = document.load_page(page_number)
            # Amplia a área de busca para garantir que capturamos todas as linhas da tabela
            words_below_headers = page.get_text(
                "words", clip=(x1_table - 20, y1_table, x2_table + 20, page.rect.height))

        # Filtra palavras que estão abaixo dos cabeçalhos e dentro dos limites horizontais da tabela
        relevant_words = []
        for word_info in words_below_headers:
            word_x0, word_y0, word_x1, word_y1, word_text = word_info[:5]
            
            # Considera apenas palavras que estão abaixo da linha de cabeçalho
            # e dentro dos limites horizontais da tabela (com uma pequena margem)
            if (word_y0 > y1_table + avg_header_height and 
                x1_table - 20 <= word_x0 <= x2_table + 20):
                relevant_words.append(word_info)
                
        # Ordena as palavras por coordenada Y para detectar saltos entre linhas
        relevant_words.sort(key=lambda w: w[1])

        # Define um valor padrão para o final da tabela
        y2_table = y1_table + avg_header_height + 50
        
        # Analisa as palavras para detectar um salto vertical significativo,
        # que pode indicar o final da tabela
        if relevant_words:
            # Agrupa palavras em linhas baseado na coordenada Y
            lines = []
            current_line = [relevant_words[0]]
            current_line_y = relevant_words[0][1]
            
            for i in range(1, len(relevant_words)):
                word_y0 = relevant_words[i][1]
                
                # Se a diferença de Y for pequena, considera como mesma linha
                if abs(word_y0 - current_line_y) <= avg_header_height / 2:
                    current_line.append(relevant_words[i])
                else:
                    # Verifica se é um salto significativo
                    if word_y0 - current_line_y > y_jump_threshold:
                        # Encontrou um salto significativo, considera o final da linha atual como o final da tabela
                        y2_table = max(w[3] for w in current_line) + 5
                        break
                    
                    # Inicia uma nova linha
                    lines.append(current_line)
                    current_line = [relevant_words[i]]
                    current_line_y = word_y0
            
            # Adiciona a última linha, se não foi adicionada no loop
            if current_line and not any(current_line[0] in line for line in lines):
                lines.append(current_line)
            
            # Se não encontrou um salto significativo, usa a última linha como referência
            if y2_table == y1_table + avg_header_height + 50 and lines:
                last_line = lines[-1]
                y2_table = max(w[3] for w in last_line) + 5
        
        table_boundary = TableBoundary(x1_table, y1_table, x2_table, y2_table)
        
        return (table_boundary, column_separators_x_coords, sorted_headers_by_x, 
                all_words_on_page, headers_coords)


class TableDataExtractor:
    """Classe responsável por extrair dados de tabelas em PDFs."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
    
    def _process_row(self, row_words: List[Any], col_boundaries: List[float]) -> List[str]:
        """
        Processa uma linha de palavras, distribuindo-as nas colunas corretas.
        
        Args:
            row_words: Lista de palavras na linha
            col_boundaries: Lista de coordenadas X dos limites das colunas
            
        Returns:
            Lista de strings, uma para cada célula da linha
        """
        # Inicializa células vazias
        row_cells = [""] * (len(col_boundaries) - 1)
        
        # Distribui palavras nas células corretas
        for word_info in row_words:
            x0, y0, x1, y1, word_text = word_info[:5]
            
            # Determina a qual coluna a palavra pertence
            for col_idx in range(len(col_boundaries) - 1):
                col_start_x = col_boundaries[col_idx]
                col_end_x = col_boundaries[col_idx + 1]
                
                # Uma palavra pertence a uma coluna se seu x0 está dentro dos limites da coluna
                # ou se ela começa antes e se estende para dentro da coluna
                if col_start_x <= x0 < col_end_x or (x0 < col_start_x and x1 > col_start_x):
                    # Adiciona o texto da palavra à célula
                    if not row_cells[col_idx]:
                        row_cells[col_idx] = word_text
                    else:
                        row_cells[col_idx] += " " + word_text
                    break
        
        # Remove espaços extras
        return [cell.strip() for cell in row_cells]
    
    def extract_table_data(self, page_number: int, table_bbox: TableBoundary, 
                          column_x_coords: List[float], header_names: List[str], 
                          y_tolerance: int = 3) -> pd.DataFrame:
        """
        Extrai os dados da tabela manualmente usando as coordenadas da área e das colunas.

        Args:
            page_number: Número da página (baseado em 0)
            table_bbox: Limites da tabela
            column_x_coords: Lista de coordenadas X dos separadores de coluna
            header_names: Lista de nomes dos cabeçalhos em ordem
            y_tolerance: Tolerância para agrupar palavras na mesma linha

        Returns:
            DataFrame contendo os dados da tabela
        """
        table_area = table_bbox.as_tuple()
        
        with PDFDocumentHandler(self.pdf_path) as document:
            page = document.load_page(page_number)
            
            # Calcula a altura média dos cabeçalhos para usar como referência
            # para separar o cabeçalho dos dados
            header_height = 0
            try:
                # Extrai texto apenas da área de cabeçalho para determinar sua altura
                header_area_words = page.get_text(
                    "words", 
                    clip=(table_area[0], table_area[1], table_area[2], table_area[1] + 30)
                )
                if header_area_words:
                    header_heights = [w[3] - w[1] for w in header_area_words]
                    header_height = sum(header_heights) / len(header_heights)
            except Exception as e:
                print(f"Aviso: Não foi possível determinar a altura do cabeçalho: {e}")
                header_height = 15  # Valor padrão
            
            # Extrai palavras da área da tabela, excluindo a linha de cabeçalho
            data_area_y1 = table_area[1] + header_height
            words_in_table_area = page.get_text(
                "words", 
                clip=(table_area[0]-5, data_area_y1, table_area[2]+5, table_area[3]+5)
            )
        
        # Filtra palavras na área de dados da tabela
        data_words = []
        for w in words_in_table_area:
            if (table_area[0] <= w[0] <= table_area[2] and 
                data_area_y1 <= w[1] <= table_area[3]):
                data_words.append(w)
        
        # Se não encontrou palavras na área de dados, tenta novamente com a área completa
        if not data_words:
            print("Aviso: Não foram encontradas palavras na área de dados. Tentando com a área completa.")
            with PDFDocumentHandler(self.pdf_path) as document:
                page = document.load_page(page_number)
                words_in_table_area = page.get_text(
                    "words", 
                    clip=(table_area[0]-5, table_area[1], table_area[2]+5, table_area[3]+5)
                )
            
            data_words = []
            for w in words_in_table_area:
                if (table_area[0] <= w[0] <= table_area[2] and 
                    table_area[1] < w[1] <= table_area[3]):  # Exclui palavras exatamente na linha y1
                    data_words.append(w)
        
        # Ordena por y0 (linha) e depois por x0 (coluna)
        data_words.sort(key=lambda w: (w[1], w[0]))
        
        # Define os limites X de cada coluna
        col_boundaries = [table_area[0]] + column_x_coords + [table_area[2]]
        
        # Inicializa estrutura para armazenar dados
        table_data = []
        current_row_y = -1
        current_row_words = []
        
        # Agrupa palavras em linhas baseado na coordenada Y
        for word_info in data_words:
            x0, y0, x1, y1, word_text = word_info[:5]
            
                        # Se é a primeira palavra ou está em uma nova linha
            if current_row_y == -1 or abs(y0 - current_row_y) > y_tolerance:
                # Processa a linha anterior, se existir
                if current_row_words:
                    row_data = self._process_row(current_row_words, col_boundaries)
                    if any(cell.strip() for cell in row_data):  # Ignora linhas vazias
                        table_data.append(row_data)
                
                # Inicia uma nova linha
                current_row_y = y0
                current_row_words = [word_info]
            else:
                # Adiciona à linha atual
                current_row_words.append(word_info)
        
        # Processa a última linha
        if current_row_words:
            row_data = self._process_row(current_row_words, col_boundaries)
            if any(cell.strip() for cell in row_data):
                table_data.append(row_data)
        
        # Cria DataFrame
        # Ajusta o número de colunas se necessário
        num_columns = len(header_names)
        for row in table_data:
            if len(row) < num_columns:
                row.extend([""] * (num_columns - len(row)))
            elif len(row) > num_columns:
                row = row[:num_columns]
        
        df = pd.DataFrame(table_data, columns=header_names)
        return df


class TableVisualizer:
    """Classe responsável por visualizar a área da tabela e seus componentes."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
    
    def visualize_table(self, page_number: int, table_bbox: TableBoundary, 
                       column_x_coords: List[float], all_words: List[Any], 
                       headers_coords: Dict[str, Tuple[float, float, float, float]], 
                       output_filename: str = "debug_table_visual.png") -> None:
        """
        Converte uma página de PDF em imagem e desenha a área da tabela, colunas e palavras.

        Args:
            page_number: Número da página (baseado em 0)
            table_bbox: Limites da tabela
            column_x_coords: Lista de coordenadas X das linhas de separação das colunas
            all_words: Lista de todas as palavras detectadas na página
            headers_coords: Dicionário com as coordenadas dos cabeçalhos encontrados
            output_filename: Nome do arquivo para salvar a imagem de depuração
        """
        table_area = table_bbox.as_tuple()
        
        with PDFDocumentHandler(self.pdf_path) as document:
            page = document.load_page(page_number)
            zoom = 2  # Aumenta a resolução para melhor visualização
            mat = fitz.Matrix(zoom, zoom)
            pix = page.get_pixmap(matrix=mat)

        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)

        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        font = cv2.FONT_HERSHEY_SIMPLEX

        # 1. Desenha todas as palavras
        for word_info in all_words:
            x0, y0, x1, y1, word_text = word_info[:5]
            scaled_x0, scaled_y0, scaled_x1, scaled_y1 = [int(coord * zoom) for coord in (x0, y0, x1, y1)]
            cv2.rectangle(img, (scaled_x0, scaled_y0), (scaled_x1, scaled_y1), (200, 200, 200), 1)

        # 2. Desenha as caixas delimitadoras dos cabeçalhos
        for phrase, bbox in headers_coords.items():
            x0, y0, x1, y1 = bbox
            scaled_x0, scaled_y0, scaled_x1, scaled_y1 = [int(coord * zoom) for coord in (x0, y0, x1, y1)]
            cv2.rectangle(img, (scaled_x0, scaled_y0), (scaled_x1, scaled_y1), (0, 255, 0), 2)
            cv2.putText(img, phrase, (scaled_x0, scaled_y0 - 5), font, 0.6, (0, 255, 0), 1, cv2.LINE_AA)

        # 3. Desenha o retângulo da área da tabela
        x1, y1, x2, y2 = [int(coord * zoom) for coord in table_area]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(img, f"Area: ({table_area[0]:.0f},{table_area[1]:.0f})-({table_area[2]:.0f},{table_area[3]:.0f})", 
                    (x1, y1 - 10), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        
        # 4. Desenha as linhas de separação das colunas
        for x_coord in column_x_coords:
            scaled_x_coord = int(x_coord * zoom)
            cv2.line(img, (scaled_x_coord, y1), (scaled_x_coord, y2), (255, 0, 0), 1)

        # Salva a imagem de depuração
        cv2.imwrite(output_filename, img)
        print(f"Imagem de depuração salva como: {output_filename}")


class PDFTableExtractor:
    """Classe principal que coordena o processo de extração de tabelas de PDFs."""
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.header_extractor = HeaderExtractor(pdf_path)
        self.table_inferer = TableAreaInferer(pdf_path, self.header_extractor)
        self.data_extractor = TableDataExtractor(pdf_path)
        self.visualizer = TableVisualizer(pdf_path)
        self.page = None  # type: ignore

    def get_table_and_page_bottom_coords(
        self,
        boundary: TableBoundary,
        page_xy_max: Tuple[float, float]
    ) -> Tuple[Tuple[float, float], Tuple[float, float]]:
        """
        Retorna:
            - a coordenada (xmin, ymax) do fim da tabela,
            - a coordenada (xmax, ymax) da largura total da página no fim da tabela.

        Parâmetros:
            linhas: lista de dicionários, cada um com 'bbox' = (x0, y0, x1, y1)
            page: objeto page do fitz

        Returns:
            ((xmin, ymax), (xmax, ymax))
        """        
        
        # x0 é início da linha, x1 é fim da linha; y1 é fim vertical da linha
        xmin = boundary.x1
        ymax = boundary.y2
        
        return (xmin, ymax), (float(page_xy_max[0]), float(page_xy_max[1]))
    
    def extract_table(self, table_config: TableConfig, 
                     visualize: bool = True,
                     visualization_output: str = "debug_table_visual.png") -> Optional[pd.DataFrame]:
        """
        Extrai uma tabela de um PDF, usando a configuração fornecida.
        
        Args:
            table_config: Configuração da tabela a ser extraída
            visualize: Se True, gera uma imagem de depuração
            visualization_output: Nome do arquivo para a imagem de depuração
            
        Returns:
            DataFrame contendo os dados da tabela ou None se falhar
        """
        # Inferir a área da tabela e as colunas
        result = self.table_inferer.infer_table_area(
            table_config.page_number, 
            table_config.header_phrases, 
            table_config.header_y_filter, 
            table_config.y_jump_threshold,
            table_config.max_header_y_variance,
            table_config.min_header_match_ratio,
            table_config.header_context_words
        )
        
        if not all(result):
            print(f"Não foi possível inferir a área da tabela '{table_config.name}' ou os separadores de coluna.")
            return None
            
        table_bbox, column_x_coords, sorted_headers_by_x, all_words, headers_coords = result
        
        with PDFDocumentHandler(self.pdf_path) as document:
            page = document.load_page(table_config.page_number)
            pdf_height = page.rect.height

            # Exemplo: suponha table_bbox é um objeto bbox com atributos x0, y0, x1, y1 pelo fitz!
            # Cuidado, PyMuPDF: y0 é topo, y1 é rodapé
            x1_c = int(table_bbox.x1)
            x2_c = int(table_bbox.x2)
            y1_c = int(pdf_height - table_bbox.y2)  # y1 tabela (menor y do fitz) -> maior y do Camelot (de baixo para cima)
            y2_c = int(0)  # y0 tabela (maior y do fitz) -> menor y do Camelot

            table_area =[f"{x1_c},{y1_c},{x2_c},{y2_c}"]       
            print(table_area)     
            
            
        
        #((x1,y1),(x2,y2))=self.get_table_and_page_bottom_coords(table_bbox,coord)
        tables = camelot.read_pdf(self.pdf_path, flavor='lattice', table_areas=table_area)    
        df = pd.DataFrame(tables[0].df)    
        print(df)
        
        
        print(f"Tabela: {table_config.name}")
        print(f"Área da Tabela Inferida: {table_bbox.as_tuple()}")
        print(f"Separadores de Coluna Inferidos: {column_x_coords}")
        print(f"Cabeçalhos encontrados: {[h[0] for h in sorted_headers_by_x]}")
        
        # Visualizar a área da tabela e as colunas, se solicitado
        if visualize:
            output_file = f"{os.path.splitext(visualization_output)[0]}_{table_config.name}{os.path.splitext(visualization_output)[1]}"
            self.visualizer.visualize_table(
                table_config.page_number, table_bbox, column_x_coords, 
                all_words, headers_coords, output_file
            )

        # Extrair os nomes dos cabeçalhos na ordem correta
        header_names_ordered = [h[0] for h in sorted_headers_by_x]

        try:
            # Extrair os dados da tabela
            df_table = self.data_extractor.extract_table_data(
                table_config.page_number, table_bbox, column_x_coords, header_names_ordered
            )
            return df_table
            
        except Exception as e:
            print(f"Ocorreu um erro ao extrair a tabela '{table_config.name}': {e}")
            return None
    
    def extract_all_tables(self, config_path: str, visualize: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Extrai todas as tabelas definidas no arquivo de configuração.
        
        Args:
            config_path: Caminho para o arquivo de configuração YAML
            visualize: Se True, gera imagens de depuração para cada tabela
            
        Returns:
            Dicionário de DataFrames, indexado pelo nome da tabela
        """
        try:
            table_configs = ConfigLoader.load_config(config_path)
        except Exception as e:
            print(f"Erro ao carregar configurações: {e}")
            return {}
        
        results = {}
        
        for table_name, table_config in table_configs.items():
            print(f"\nProcessando tabela: {table_name}")
            df = self.extract_table(
                table_config,
                visualize=visualize,
                visualization_output=f"debug_{table_name}.png"
            )
            
            if df is not None:
                results[table_name] = df
                print(f"Tabela '{table_name}' extraída com sucesso.")
            else:
                print(f"Falha ao extrair tabela '{table_name}'.")
        
        return results


def create_example_config(output_path: str = "table_configs.yaml") -> None:
    """
    Cria um arquivo de configuração YAML de exemplo.
    
    Args:
        output_path: Caminho para salvar o arquivo de configuração
    """
    example_config = {
        'tables': {
            'xp_negociacao': {
                'header_phrases': [
                    "Negociação", "C/V", "Tipo mercado", "Prazo",
                    "Especificação do título", "Obs. (*)", "Quantidade",
                    "Preço / Ajuste", "Valor Operação / Ajuste", "D/C"
                ],
                'header_y_filter': {
                    'min': None,
                    'max': None
                },
                'y_jump_threshold': 15.0,
                'page_number': 0,
                'max_header_y_variance': 5.0,
                'min_header_match_ratio': 0.7,
                'header_context_words': ["negociação", "título", "mercado", "quantidade"]
            },
            'xp_resumo_financeiro': {
                'header_phrases': [
                    "Resumo Financeiro", "Valor", "Resumo Negócios", "Valor"
                ],
                'header_y_filter': {
                    'min': 700,
                    'max': 750
                },
                'y_jump_threshold': 15.0,
                'page_number': 0,
                'max_header_y_variance': 5.0,
                'min_header_match_ratio': 0.7,
                'header_context_words': ["resumo", "financeiro", "negócios", "total"]
            }
        }
    }
    
    with open(output_path, 'w', encoding='utf-8') as file:
        yaml.dump(example_config, file, default_flow_style=False, allow_unicode=True, sort_keys=False)
    
    print(f"Arquivo de configuração de exemplo criado: {output_path}")


def main():
    """Função principal de demonstração."""
    # Verifica se o arquivo de configuração existe, caso contrário cria um exemplo
    config_file = "table_configs.yaml"
    if not os.path.exists(config_file):
        create_example_config(config_file)
        print(f"Criado arquivo de configuração de exemplo: {config_file}")
        print("Por favor, ajuste o arquivo de configuração conforme necessário e execute novamente.")
        return
    
    pdf_file = "nubank.pdf"
    
    # Verifica se o arquivo PDF existe
    if not os.path.exists(pdf_file):
        print(f"Arquivo PDF não encontrado: {pdf_file}")
        print("Por favor, ajuste o caminho do arquivo PDF no código ou coloque o arquivo no diretório correto.")
        return
    
    extractor = PDFTableExtractor(pdf_file)
    
    # Extrai todas as tabelas definidas no arquivo de configuração
    tables = extractor.extract_all_tables(config_file)
    
    # Exibe os resultados
    for table_name, df in tables.items():
        print(f"\nDados da Tabela '{table_name}':")
        print(df)
        
        # Salva o DataFrame em um arquivo CSV
        csv_file = f"{table_name}.csv"
        df.to_csv(csv_file, index=False, encoding='utf-8-sig')
        print(f"Dados salvos em: {csv_file}")


if __name__ == "__main__":
    main()