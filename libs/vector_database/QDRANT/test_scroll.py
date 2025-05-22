from qdrant_client import QdrantClient, models
from qdrant_client.models import Distance, VectorParams, PointStruct

time_in = 20000
time_out = 25000

client = QdrantClient(url="http://localhost:6333")
FILTER_RESULTS = models.Filter(
                    must = [
                        models.FieldCondition(
                            key="video_name",
                            match=models.MatchValue(value="L22_V025" + '.mp4'),
                        ),
                        models.FieldCondition(
                            key="frame_name",
                            range=models.Range(
                                gte = None if time_in == "" else int(time_in),
                                lte = None if time_out == "" else int(time_out)
                            ),
                        ),
                    ]
                )
SCROLL_RESULT = client.scroll(
    collection_name = "MOBILE",
    scroll_filter=FILTER_RESULTS,
    with_payload=True,
    with_vectors=False
)

for item in SCROLL_RESULT[0]:
    for idx, field in enumerate(item):
        if idx == 0:
            key = str(field[1])
        elif idx == 1:
            idx_folder = str(field[1]['idx_folder'])
            video_name = str(field[1]['video_name'])
            keyframe_id = str(field[1]['frame_name']).zfill(5)
    print(key, idx_folder, video_name, keyframe_id)