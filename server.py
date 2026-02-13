import os
import asyncio
import json
import tempfile
import re
from typing import Optional, List, Dict
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification
import torch
import logging
from collections import deque
from datetime import datetime
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="LocalMeeting AI Server - Advanced Edition")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODELS_DIR = "./models"
os.makedirs(MODELS_DIR, exist_ok=True)

logger.info("Loading Neural Networks with Advanced AI Techniques...")

device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")

try:
    transcriber = WhisperModel("medium", device="auto", compute_type="int8")
    logger.info("Whisper model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Whisper model: {e}")
    transcriber = None

try:
    sentiment_analyzer = pipeline(
        "sentiment-analysis",
        model="nlptown/bert-base-multilingual-uncased-sentiment",
        device=0 if device == "cuda" else -1
    )
    logger.info("Sentiment analyzer loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentiment analyzer: {e}")
    sentiment_analyzer = None

summarizer = None
emotion_classifier = None
text_generator = None

try:
    summarizer = pipeline(
        "summarization",
        model="facebook/bart-large-cnn",
        device=0 if device == "cuda" else -1
    )
    logger.info("Summarizer loaded successfully")
except Exception as e:
    logger.warning(f"Summarizer not available: {e}")

try:
    emotion_classifier = pipeline(
        "text-classification",
        model="j-hartmann/emotion-english-distilroberta-base",
        device=0 if device == "cuda" else -1,
        top_k=None
    )
    logger.info("Emotion classifier loaded successfully")
except Exception as e:
    logger.warning(f"Emotion classifier not available: {e}")

try:
    text_generator = pipeline(
        "text-generation",
        model="gpt2",
        device=0 if device == "cuda" else -1
    )
    logger.info("Text generator loaded successfully")
except Exception as e:
    logger.warning(f"Text generator not available: {e}")

class AdvancedTextAnalyzer:
    def __init__(self):
        self.keyword_extractor = self._init_keyword_extractor()
        
    def _init_keyword_extractor(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            return TfidfVectorizer(max_features=10, stop_words='english', ngram_range=(1, 2))
        except:
            return None
    
    def extract_keywords(self, text: str) -> List[str]:
        if not self.keyword_extractor or len(text.split()) < 10:
            return []
        
        try:
            tfidf_matrix = self.keyword_extractor.fit_transform([text])
            feature_names = self.keyword_extractor.get_feature_names_out()
            return list(feature_names[:5])
        except:
            return []
    
    def extract_action_items(self, text: str) -> List[str]:
        action_patterns = [
            r'(?:precisa(?:mos)?|deve(?:mos)?|vai(?:mos)?|tem que)\s+(.+?)(?:\.|,|$)',
            r'(?:vou|vamos|iremos)\s+(.+?)(?:\.|,|$)',
            r'(?:fazer|criar|desenvolver|implementar|revisar|analisar)\s+(.+?)(?:\.|,|$)',
        ]
        
        actions = []
        for pattern in action_patterns:
            matches = re.findall(pattern, text.lower())
            actions.extend(matches[:3])
        
        return list(set(actions))[:5]
    
    def extract_decisions(self, text: str) -> List[str]:
        decision_patterns = [
            r'(?:decidimos|decidiu-se|ficou decidido|concordamos)\s+(?:que\s+)?(.+?)(?:\.|,|$)',
            r'(?:aprovado|rejeitado|aceito)\s+(.+?)(?:\.|,|$)',
        ]
        
        decisions = []
        for pattern in decision_patterns:
            matches = re.findall(pattern, text.lower())
            decisions.extend(matches)
        
        return list(set(decisions))[:5]
    
    def detect_questions(self, text: str) -> List[str]:
        sentences = text.split('.')
        questions = [s.strip() + '?' for s in sentences if '?' in s or any(q in s.lower() for q in ['como', 'quando', 'onde', 'quem', 'qual', 'quanto'])]
        return questions[:5]

class MeetingProcessor:
    def __init__(self):
        self.transcript_accumulator = []
        self.audio_buffer = bytearray()
        self.buffer_limit = 250000
        self.chunk_count = 0
        self.analyzer = AdvancedTextAnalyzer()
        self.sentiment_history = deque(maxlen=20)
        self.emotion_history = deque(maxlen=20)
        self.start_time = datetime.now()

    def analyze_sentiment(self, text: str) -> str:
        if not sentiment_analyzer or len(text.split()) < 3:
            return "Neutro"
        
        try:
            result = sentiment_analyzer(text[:512])[0]
            score = int(result['label'].split()[0])
            
            self.sentiment_history.append(score)
            
            if score >= 4:
                return "Positivo"
            if score <= 2:
                return "Negativo"
            return "Neutro"
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return "Neutro"
    
    def analyze_emotions(self, text: str) -> Dict:
        if not emotion_classifier:
            return {}
        
        try:
            results = emotion_classifier(text[:512])[0]
            top_emotion = max(results, key=lambda x: x['score'])
            self.emotion_history.append(top_emotion['label'])
            return {
                'primary': top_emotion['label'],
                'confidence': round(top_emotion['score'], 2)
            }
        except Exception as e:
            logger.error(f"Emotion analysis error: {e}")
            return {}

    def transcribe(self, audio_data: bytes) -> str:
        if not transcriber:
            logger.error("Transcriber not available")
            return ""
        
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp:
            tmp.write(audio_data)
            tmp_path = tmp.name

        try:
            segments, _ = transcriber.transcribe(
                tmp_path,
                language="pt",
                beam_size=5,
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,
                    speech_pad_ms=400
                )
            )
            text = " ".join([s.text.strip() for s in segments]).strip()
            return text
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return ""
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except Exception as e:
                    logger.error(f"Error removing temp file: {e}")

    def generate_summary(self, text: str) -> str:
        if not summarizer or len(text.split()) < 50:
            return text[:500] + "..." if len(text) > 500 else text
        
        try:
            max_length = min(200, len(text.split()) // 2)
            min_length = min(50, len(text.split()) // 4)
            
            summary = summarizer(
                text,
                max_length=max_length,
                min_length=min_length,
                do_sample=False
            )[0]['summary_text']
            
            return summary
        except Exception as e:
            logger.error(f"Summarization error: {e}")
            return text[:500] + "..."

    def calculate_meeting_stats(self) -> Dict:
        duration = (datetime.now() - self.start_time).total_seconds()
        
        avg_sentiment = sum(self.sentiment_history) / len(self.sentiment_history) if self.sentiment_history else 3
        
        sentiment_label = "Neutro"
        if avg_sentiment >= 4:
            sentiment_label = "Positivo"
        elif avg_sentiment <= 2:
            sentiment_label = "Negativo"
        
        emotion_counts = {}
        for emotion in self.emotion_history:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else "neutral"
        
        return {
            'duration_minutes': round(duration / 60, 1),
            'total_segments': len(self.transcript_accumulator),
            'avg_sentiment': sentiment_label,
            'dominant_emotion': dominant_emotion,
            'sentiment_trend': 'improving' if len(self.sentiment_history) > 5 and list(self.sentiment_history)[-3:] > list(self.sentiment_history)[:3] else 'stable'
        }

    def generate_report(self, full_text: str) -> str:
        if not full_text or len(full_text.strip()) < 50:
            return "Erro: Transcrição insuficiente para gerar relatório. Certifique-se de capturar pelo menos 30 segundos de áudio."

        try:
            keywords = self.analyzer.extract_keywords(full_text)
            action_items = self.analyzer.extract_action_items(full_text)
            decisions = self.analyzer.extract_decisions(full_text)
            questions = self.analyzer.detect_questions(full_text)
            stats = self.calculate_meeting_stats()
            
            summary = self.generate_summary(full_text)
            
            report_parts = []
            
            report_parts.append("=" * 60)
            report_parts.append("RELATÓRIO DE REUNIÃO - ANÁLISE AVANÇADA COM IA")
            report_parts.append("=" * 60)
            report_parts.append("")
            
            report_parts.append("📊 ESTATÍSTICAS DA REUNIÃO")
            report_parts.append("-" * 60)
            report_parts.append(f"Duração: {stats['duration_minutes']} minutos")
            report_parts.append(f"Segmentos transcritos: {stats['total_segments']}")
            report_parts.append(f"Sentimento médio: {stats['avg_sentiment']}")
            report_parts.append(f"Emoção dominante: {stats['dominant_emotion']}")
            report_parts.append(f"Tendência: {stats['sentiment_trend']}")
            report_parts.append("")
            
            report_parts.append("📝 RESUMO EXECUTIVO")
            report_parts.append("-" * 60)
            report_parts.append(summary)
            report_parts.append("")
            
            if keywords:
                report_parts.append("🔑 TÓPICOS PRINCIPAIS")
                report_parts.append("-" * 60)
                for i, kw in enumerate(keywords, 1):
                    report_parts.append(f"{i}. {kw}")
                report_parts.append("")
            
            if decisions:
                report_parts.append("✅ DECISÕES TOMADAS")
                report_parts.append("-" * 60)
                for i, decision in enumerate(decisions, 1):
                    report_parts.append(f"{i}. {decision.capitalize()}")
                report_parts.append("")
            
            if action_items:
                report_parts.append("📋 AÇÕES FUTURAS (Action Items)")
                report_parts.append("-" * 60)
                for i, action in enumerate(action_items, 1):
                    report_parts.append(f"[ ] {i}. {action.capitalize()}")
                report_parts.append("")
            
            if questions:
                report_parts.append("❓ QUESTÕES LEVANTADAS")
                report_parts.append("-" * 60)
                for i, question in enumerate(questions, 1):
                    report_parts.append(f"{i}. {question}")
                report_parts.append("")
            
            report_parts.append("💭 ANÁLISE DO TOM DA DISCUSSÃO")
            report_parts.append("-" * 60)
            
            if stats['avg_sentiment'] == 'Positivo':
                report_parts.append("A reunião teve um tom predominantemente positivo, com")
                report_parts.append("engajamento construtivo entre os participantes.")
            elif stats['avg_sentiment'] == 'Negativo':
                report_parts.append("A reunião apresentou momentos de tensão ou preocupação,")
                report_parts.append("sugerindo possíveis desafios ou divergências de opinião.")
            else:
                report_parts.append("A reunião manteve um tom neutro e profissional, focado")
                report_parts.append("na discussão objetiva dos tópicos apresentados.")
            
            report_parts.append("")
            report_parts.append(f"Emoção predominante detectada: {stats['dominant_emotion']}")
            report_parts.append("")
            
            report_parts.append("=" * 60)
            report_parts.append("Relatório gerado automaticamente via IA Local")
            report_parts.append(f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
            report_parts.append("=" * 60)
            
            return "\n".join(report_parts)
            
        except Exception as e:
            logger.error(f"Report generation error: {e}")
            return f"Erro ao gerar relatório: {str(e)}\n\nTranscrição completa:\n{full_text}"

@app.websocket("/ws/meeting")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    processor = MeetingProcessor()
    logger.info("New WebSocket connection established")

    try:
        while True:
            data = await websocket.receive()

            if "bytes" in data:
                processor.audio_buffer.extend(data["bytes"])

                if len(processor.audio_buffer) > processor.buffer_limit:
                    processor.chunk_count += 1
                    chunk_to_process = bytes(processor.audio_buffer)
                    processor.audio_buffer = bytearray()

                    logger.info(f"Processing audio chunk {processor.chunk_count} ({len(chunk_to_process)} bytes)")

                    text = await asyncio.to_thread(processor.transcribe, chunk_to_process)

                    if text:
                        sentiment = await asyncio.to_thread(processor.analyze_sentiment, text)
                        emotions = await asyncio.to_thread(processor.analyze_emotions, text)
                        processor.transcript_accumulator.append(text)

                        logger.info(f"Transcribed: {text[:100]}... | Sentiment: {sentiment}")

                        await websocket.send_json({
                            "type": "realtime",
                            "text": text,
                            "sentiment": sentiment,
                            "emotions": emotions
                        })

            elif "text" in data:
                msg = json.loads(data["text"])
                if msg.get("command") == "generate_report":
                    logger.info("Generating advanced AI report...")
                    full_text = " ".join(processor.transcript_accumulator)

                    if full_text:
                        report = await asyncio.to_thread(processor.generate_report, full_text)

                        await websocket.send_json({
                            "type": "report",
                            "content": report
                        })
                        logger.info("Report sent successfully")
                    else:
                        await websocket.send_json({
                            "type": "report",
                            "content": "Nenhuma transcrição foi capturada. Certifique-se de que há áudio sendo reproduzido na aba."
                        })
                    break

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        try:
            await websocket.close()
        except:
            pass

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "whisper": transcriber is not None,
        "sentiment": sentiment_analyzer is not None,
        "summarizer": summarizer is not None,
        "emotion": emotion_classifier is not None,
        "text_gen": text_generator is not None,
        "device": device
    }

if __name__ == "__main__":
    import uvicorn
    logger.info("Starting LocalMeeting AI Server (Advanced Edition) on http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")