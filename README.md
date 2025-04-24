# CGM: Code Graph LLM

![CodefuseLogo](./assets/github-codefuse-logo-update.jpg)

## Contents
- [News](#News)
- [Introduction](#Introduction)
- [Contributing](#Contributing)
- [Join Us](#Join-Us)

## News

🔥🔥🔥 [2025/01/15] We are pleased to announce the updated version of the CGM-72B-V1.2. The model further achieves a remarkable 43.00% resolve rate on the SWE-Bench-Lite leaderboard.

🔥🔥🔥 [2024/12/28] We are pleased to announce the updated version of the CGM-72B-V1.1. The model further achieves a remarkable 41.67% resolve rate on the SWE-Bench-Lite leaderboard.

🔥🔥🔥 [2024/10/28] We are pleased to announce the release of the CGM-72B. The model achieves a remarkable 35.67% resolve rate on the SWE-Bench-Lite leaderboard.

🔥🔥🔥 [2024/10/28] We released **CGM**, mainly for repo-level coding tasks.

## Introduction
We propose a graph-based framework CGM for real-world SE tasks. Before CGM starts its work, we construct a repository-level code graph to better represent the repository context and its structure by Code Graph Generator. Inspired by the Retrieval-Augmented Generation (RAG) approach, CGM framework is designed as a chain structure consisting of four atomic nodes, termed as R4 (Rewriter, Retriever, Reranker, and Reader) chain for this scenario. Given an issue, the initial input to the CGM framework includes the issue description and the corresponding code graph. Rewriter will first rewrite the original issue by extracting keywords and generating relevant queries for code graph. Then a heuristic code subgraph is retrieved through Retriever based on the matching anchor nodes from rewriter output. Given that the resulting subgraph provides a relatively broad context necessary for reference, we need a Reranker to identify the files most likely to be modified as a further hint. Subsequently, both the retrieved subgraph and the identified files are input into a trainable, graph-based Reader to generate the corresponding code patch.

### Framework

![Framework](./assets/cgm_framework_0123.png)

### Highlights
:white_check_mark: **Code Graph**: Train models on multiple tasks while maintaining a balance between them. The models can even generalize to new, previously unseen tasks.

:white_check_mark: **Multi-framework**: It provides support for both Accelerate (with Deepspeed and FSDP)

:white_check_mark: **Efficient fine-tuning**: It supports LoRA, QLoRA as well as Full-parameters training, enabling fine-tuning of large models with minimal resources. The training speed meets the demands of almost all fine-tuning scenarios.

## Contributing
Contributions are welcome! If you have any suggestions, ideas, bug reports, or new model/feature supported, please open an issue or submit a pull request.

## Citation
If you find our work useful or helpful for your R&D works, please feel free to cite our paper as below.

## Join-US

We are the AI Native team within the Platform Technology Business Group at Ant Group, dedicated to the intelligentization of Ant Group's platform engineering. Established for over three years, our team has played a pivotal role in supporting the intelligent operation and maintenance of Ant Group's cloud computing infrastructure. Our mission is to build algorithm services and platforms with a wide user base through world-class technological innovation and impact, supporting the implementation of internal and external products and businesses.
Embracing an innovation-driven ethos, our team not only supports business implementation but also propels technological influence. Over the past three years, we have published more than 20 papers at top conferences like ICLR, NeurIPS, KDD, and ACL. Our innovative business outcomes have earned us two Ant Technology's highest T-Star awards and one SuperMA award from Ant Group. Our open-source project CodeFuse has received 4K stars as of February 2024, and our models have been downloaded over 1.5 million times on Huggingface and Modelscope.

We are on the lookout for top talents to join our vibrant team! If you're eager to develop your career in an environment filled with energy, innovation, and a culture of excellence, we welcome you to explore our career opportunities for both campus and experienced hires. Join us and be a part of creating the next milestone in the industry.

**Contact**: hyu.hugo@antgroup.com 
