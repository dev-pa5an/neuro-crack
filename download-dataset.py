import asyncio
import skillsnetwork

async def main():
    await skillsnetwork.prepare("https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0321EN/data/images/concrete_crack_images_for_classification.zip", path="resources/data", overwrite=True)

asyncio.run(main())
