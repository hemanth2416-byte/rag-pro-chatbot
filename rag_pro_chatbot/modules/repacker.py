def repack(docs, strategy="reverse"):
    if strategy == "reverse":
        return "\n\n".join(docs[::-1])
    return "\n\n".join(docs)

