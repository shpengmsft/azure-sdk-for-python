FROM mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:20230808.v1

ENV AZUREML_CONDA_ENVIRONMENT_PATH /azureml-envs/myenv
# Create conda environment
RUN conda create -p $AZUREML_CONDA_ENVIRONMENT_PATH \
    python=3.8 pip=21.3.1 -c anaconda -c conda-forge

# Prepend path to AzureML conda environment
ENV PATH $AZUREML_CONDA_ENVIRONMENT_PATH/bin:$PATH

# Sentence Transformers CPU only install
RUN pip install --no-cache-dir torch==2.0.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install --no-cache-dir transformers && \
    pip install --no-cache-dir tqdm numpy scikit-learn scipy nltk sentencepiece pillow && \
    pip install --no-cache-dir sentence-transformers --no-deps

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends \
    # utilities for keeping Debian and OpenJDK CA certificates in sync
    ca-certificates p11-kit wget \
    ; \
    rm -rf /var/lib/apt/lists/*

ENV JAVA_HOME /usr/local/openjdk-21
ENV PATH $JAVA_HOME/bin:$PATH

# Default to UTF-8 file.encoding
ENV LANG C.UTF-8

# https://jdk.java.net/
# >
# > Java Development Kit builds, from Oracle
# >
ENV JAVA_VERSION 21-ea+22

RUN set -eux; \
	\
	arch="$(dpkg --print-architecture)"; \
	case "$arch" in \
		'amd64') \
			downloadUrl='https://download.java.net/java/early_access/jdk21/34/GPL/openjdk-21-ea+34_linux-x64_bin.tar.gz'; \
			downloadSha256='d16356385a726320077563c6180f2eabf72ded64b5695f24f3e7a2d3980b1f11'; \
			;; \
		*) echo >&2 "error: unsupported architecture: '$arch'"; exit 1 ;; \
	esac; \
	\
	savedAptMark="$(apt-mark showmanual)"; \
	\
	wget --progress=dot:giga -O openjdk.tgz "$downloadUrl"; \
	echo "$downloadSha256 *openjdk.tgz" | sha256sum --strict --check -; \
	\
	mkdir -p "$JAVA_HOME"; \
	tar --extract \
		--file openjdk.tgz \
		--directory "$JAVA_HOME" \
		--strip-components 1 \
		--no-same-owner \
	; \
	rm openjdk.tgz*; \
	\
	apt-mark auto '.*' > /dev/null; \
	[ -z "$savedAptMark" ] || apt-mark manual $savedAptMark > /dev/null; \
	apt-get purge -y --auto-remove -o APT::AutoRemove::RecommendsImportant=true; \
	\
    # update "cacerts" bundle to use Debian's CA certificates (and make sure it stays up-to-date with changes to Debian's store)
    # see https://github.com/docker-library/openjdk/issues/327
    #     http://rabexc.org/posts/certificates-not-working-java#comment-4099504075
    #     https://salsa.debian.org/java-team/ca-certificates-java/blob/3e51a84e9104823319abeb31f880580e46f45a98/debian/jks-keystore.hook.in
    #     https://git.alpinelinux.org/aports/tree/community/java-cacerts/APKBUILD?id=761af65f38b4570093461e6546dcf6b179d2b624#n29
	mkdir -p /etc/ca-certificates/update.d; \
    ls -al /etc/ca-certificates; \
    { \
		echo '#!/usr/bin/env bash'; \
		echo 'set -Eeuo pipefail'; \
		echo 'trust extract --overwrite --format=java-cacerts --filter=ca-anchors --purpose=server-auth "$JAVA_HOME/lib/security/cacerts"'; \
	} > /etc/ca-certificates/update.d/docker-openjdk; \
	chmod +x /etc/ca-certificates/update.d/docker-openjdk; \
	/etc/ca-certificates/update.d/docker-openjdk; \
    update-ca-certificates; \
	\
    # https://github.com/docker-library/openjdk/issues/331#issuecomment-498834472
	find "$JAVA_HOME/lib" -name '*.so' -exec dirname '{}' ';' | sort -u > /etc/ld.so.conf.d/docker-openjdk.conf; \
	ldconfig; \
	\
    # https://github.com/docker-library/openjdk/issues/212#issuecomment-420979840
    # https://openjdk.java.net/jeps/341
	java -Xshare:dump; \
	\
    # basic smoke test
	fileEncoding="$(echo 'System.out.println(System.getProperty("file.encoding"))' | jshell -s -)"; [ "$fileEncoding" = 'UTF-8' ]; rm -rf ~/.java; \
	javac --version; \
	java --version

# Install azureml-rag to get all deps then uninstall it because snapshot will have azureml-rag source.
RUN pip install --no-cache-dir azureml-rag[faiss,cognitive_search,document_parsing]>=0.1.20 && \
	pip uninstall azureml-rag -y
# Install from local for wheel based testing (2 steps)
# 1. Generate wheel in cmd prompt
# > d:\src\AzureMlCli\src\azureml-rag>python setup.py bdist_wheel & copy dist\azureml_rag-0.1.0.0-py3-none-any.whl tests\remote\environments\
# 2. Uncomment next 2 lines Copy & Run
# COPY azureml_rag-0.1.0.0-py3-none-any.whl .
# RUN pip install azureml_rag-0.1.0.0-py3-none-any.whl[faiss,cognitive_search,document_parsing] && rm -rf azureml_rag-0.1.0.0-py3-none-any.whl
RUN pip install --no-cache-dir 'psutil~=5.8.0' \
    'pandas~=1.1.5' \
    'GitPython>=3.1' \
    "msrest>=0.6.18" \
    "azureml-telemetry" \
    "azure-core<2.0.0,>=1.8.0, !=1.22.0" \
    "azure-mgmt-core<2.0.0,>=1.3.0" \
    "azure-keyvault-secrets==4.6.0" \
    # package dependency needed for aml run token authentication
    'python-dateutil>=2.7.3,<3.0.0' \
    'cryptography>=41.0.3' \
    'PyJWT<3.0.0' \
    'pytz' \
	'mlflow<2.6.0'
# # TODO: Add to setup.py
# 'opentelemetry-api' \
# 'azure-monitor-opentelemetry' \
