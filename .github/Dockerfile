FROM nvcr.io/nvidia/pytorch:24.07-py3
ADD ./hidet /workspace/hidet
ADD ./models /workspace/models
WORKDIR /workspace
RUN pip install -r hidet/requirements.txt && \
    pip install -r hidet/requirements-dev.txt && \
    pip install -r hidet/.github/requirements-ci.txt && \
    bash hidet/scripts/wheel/build_wheel.sh && \
    WHEEL=$(find hidet/scripts/wheel/built_wheel -maxdepth 1 -name '*.whl') && \
    pip install --no-deps --force-reinstall $WHEEL && \
    pip install -e models && \
    hidet cache clear --all
