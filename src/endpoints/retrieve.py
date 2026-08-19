from fastapi import APIRouter

retrieve_router = APIRouter()


@retrieve_router.get("")
def retrieve():
    pass
