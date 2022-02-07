import src.model.model_cifa10.model_hyper_params as mhp

COLAB_BATCH_SIZE = 30

def test_batchsize():
    assert mhp.batch_size >= COLAB_BATCH_SIZE, f"Production batch size should be {COLAB_BATCH_SIZE}"

if __name__ == "__main__":
    test_batchsize()

    print("EVERYTHING PASSED")