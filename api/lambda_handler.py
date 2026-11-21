# AWS Lambda entry point for the Healthcare Data Quality FastAPI application.
from mangum import Mangum
from api.main import app

handler = Mangum(app, lifespan="off")
