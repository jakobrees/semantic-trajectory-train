wget https://msmarco.z22.web.core.windows.net/msmarcoranking/queries.tar.gz
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/collection.tar.gz
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/qrels.dev.tsv
wget https://msmarco.z22.web.core.windows.net/msmarcoranking/top1000.dev.tar.gz

tar -xzvf top1000.dev.tar.gz
tar -xzvf queries.tar.gz
tar -xzvf collection.tar.gz