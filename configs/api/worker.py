import os
import sys
import threading
from loguru import logger

# Ensure src path is correctly configured
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from celery import Celery
    CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")

    celery_app = Celery(
        "cyberheal_tasks",
        broker=CELERY_BROKER_URL,
        backend=CELERY_RESULT_BACKEND
    )

    celery_app.conf.update(
        task_serializer='json',
        accept_content=['json'], 
        result_serializer='json',
        timezone='UTC',
        enable_utc=True,
    )

    @celery_app.task(bind=True, max_retries=3)
    def execute_pipeline_task(self, incident_id: str):
        """
        Celery task that executes the fully autonomous LangGraph Multi-Agent pipeline.
        """
        logger.info(f"Celery Worker Task: Starting Pipeline for Incident {incident_id}")
        try:
            from api.ingestion import run_pipeline_thread
            run_pipeline_thread(incident_id)
            return {"status": "success", "incident_id": incident_id}
        except Exception as e:
            logger.error(f"Celery execution error for {incident_id}: {str(e)}")
            self.retry(exc=e, countdown=10)

except ImportError:
    class DummyCeleryTask:
        def delay(self, incident_id: str):
            logger.info(f"Threading Worker Fallback: Starting Pipeline for Incident {incident_id}")
            from api.ingestion import run_pipeline_thread
            threading.Thread(target=run_pipeline_thread, args=(incident_id,)).start()

    execute_pipeline_task = DummyCeleryTask()
    celery_app = None
