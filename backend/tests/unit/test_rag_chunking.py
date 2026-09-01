from app.rag.service import split_text


def test_sentence_aware_chunking_does_not_cut_normal_sentences():
    sentences = [
        f"第{index}项审核流程应指定审核岗用户，并完成对应审批配置。"
        for index in range(60)
    ]
    source = "\n".join(sentences)

    chunks = split_text(source, chunk_size=180, overlap=40)

    assert len(chunks) > 1
    assert all(chunk.endswith("。") for chunk in chunks)
    assert all(sentence in "\n".join(chunks) for sentence in sentences)


def test_short_section_heading_and_explanation_stay_together():
    source = (
        "3.1.1 审核流程配置\n"
        "财政用户通过系统管理、审批流程管理菜单新增审核流程。\n"
        "增加流程审批阶段，指定审核岗用户。"
    )

    assert split_text(source) == [source]
