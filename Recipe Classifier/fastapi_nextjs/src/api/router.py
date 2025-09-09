"""All API routes used in the application"""

from fastapi import APIRouter

from oma_recipeclassifier.src.api.endpoints import predict, root

api_router = APIRouter()
api_router.include_router(root.router, tags=["root"])
api_router.include_router(predict.router)
