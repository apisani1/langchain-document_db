class IDsDB():
    def __init__(self):
        self.ids_db = {}

    def add_ids(self, key: str, ids: list[str], namespace: str = ""):
        if ids and isinstance(ids, list):
            key = f"{namespace}:{key}"
            existing_ids= self.ids_db.get(key, [])
            existing_ids.extend(ids)
            self.ids_db[key] = existing_ids

    def get_ids(self, key: str, namespace: str = "") -> list[str]:
        key = f"{namespace}:{key}"
        return self.ids_db.get(key, [])

    def delete_ids(self, key: str, namespace: str = ""):
        key = f"{namespace}:{key}"
        self.ids_db.pop(key, None)
