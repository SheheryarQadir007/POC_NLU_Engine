def merge_bio_tags(tokens, labels):
    entities = []
    current_type = None
    current_tokens = []

    for token, label in zip(tokens, labels):
        if label == "O":
            if current_type is not None:
                entities.append((current_type, " ".join(current_tokens)))
                current_type = None
                current_tokens = []
            continue

        prefix, entity_type = label.split("-", 1)

        if prefix == "B":
            if current_type is not None:
                entities.append((current_type, " ".join(current_tokens)))
            current_type = entity_type
            current_tokens = [token]
        elif prefix == "I" and current_type == entity_type:
            current_tokens.append(token)
        else:
            if current_type is not None:
                entities.append((current_type, " ".join(current_tokens)))
            current_type = entity_type
            current_tokens = [token]

    if current_type is not None:
        entities.append((current_type, " ".join(current_tokens)))

    return entities