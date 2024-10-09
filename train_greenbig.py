import hashlib
import gzip
import os
import json
import lightning
import sys
import torch
import transformers

import bigbro

from transformers import logging as transformers_logging

# Set the logging level to ERROR to suppress INFO and WARNING messages
transformers_logging.set_verbosity(transformers_logging.CRITICAL)

DEBUG = False
MAX_EPOCHS = 1
TRAIN_BATCH_SIZE = 4
VALIDATION_BATCH_SIZE = 48


species = {
    "galdieria_sulphuraria": 0,
    "vitis_vinifera": 1,
    "chlamydomonas_reinhardtii": 2,
    "corchorus_capsularis": 3,
    "triticum_aestivum_mattis": 4,
    "physcomitrium_patens": 5,
    "theobroma_cacao": 6,
    "eutrema_salsugineum": 7,
    "populus_trichocarpa": 8,
    "chara_braunii": 9,
    "amborella_trichopoda": 10,
    "oryza_glaberrima": 11,
    "panicum_hallii": 12,
    "quercus_lobata": 13,
    "sesamum_indicum": 14,
    "setaria_viridis": 15,
    "cannabis_sativa_female": 16,
    "solanum_tuberosum_rh8903916": 17,
    "arabidopsis_lyrata": 18,
    "trifolium_pratense": 19,
    "triticum_aestivum_landmark": 20,
    "oryza_rufipogon": 21,
    "brassica_rapa": 22,
    "prunus_avium": 23,
    "arabidopsis_halleri": 24,
    "oryza_brachyantha": 25,
    "leersia_perrieri": 26,
    "actinidia_chinensis": 27,
    "triticum_dicoccoides": 28,
    "brassica_rapa_ro18": 29,
    "oryza_indica": 30,
    "marchantia_polymorpha": 31,
    "corylus_avellana": 32,
    "triticum_aestivum_cadenza": 33,
    "triticum_aestivum_robigus": 34,
    "setaria_italica": 35,
    "triticum_aestivum_norin61": 36,
    "oryza_nivara": 37,
    "vigna_radiata": 38,
    "triticum_aestivum_stanley": 39,
    "triticum_aestivum_julius": 40,
    "zea_mays": 41,
    "prunus_persica": 42,
    "juglans_regia": 43,
    "triticum_aestivum": 44,
    "kalanchoe_fedtschenkoi": 45,
    "triticum_aestivum_arinalrfor": 46,
    "lactuca_sativa": 47,
    "ostreococcus_lucimarinus": 48,
    "ananas_comosus": 49,
    "theobroma_cacao_criollo": 50,
    "oryza_meridionalis": 51,
    "arabis_alpina": 52,
    "cucumis_sativus": 53,
    "oryza_glumipatula": 54,
    "prunus_dulcis": 55,
    "solanum_lycopersicum": 56,
    "nymphaea_colorata": 57,
    "selaginella_moellendorffii": 58,
    "capsicum_annuum": 59,
    "gossypium_raimondii": 60,
    "triticum_aestivum_mace": 61,
    "triticum_turgidum": 62,
    "camelina_sativa": 63,
    "olea_europaea_sylvestris": 64,
    "medicago_truncatula": 65,
    "olea_europaea": 66,
    "dioscorea_rotundata": 67,
    "citrus_clementina": 68,
    "triticum_aestivum_paragon": 69,
    "cynara_cardunculus": 70,
    "helianthus_annuus": 71,
    "musa_acuminata": 72,
    "hordeum_vulgare": 73,
    "brassica_oleracea": 74,
    "aegilops_tauschii": 75,
    "triticum_aestivum_jagger": 76,
    "oryza_sativa": 77,
    "eucalyptus_grandis": 78,
    "triticum_spelta": 79,
    "chenopodium_quinoa": 80,
    "ipomoea_triloba": 81,
    "malus_domestica_golden": 82,
    "triticum_urartu": 83,
    "brachypodium_distachyon": 84,
    "triticum_aestivum_claire": 85,
    "beta_vulgaris": 86,
    "eragrostis_tef": 87,
    "triticum_aestivum_weebil": 88,
    "arabidopsis_thaliana": 89,
    "brassica_napus": 90,
    "coffea_canephora": 91,
    "oryza_barthii": 92,
    "panicum_hallii_fil2": 93,
    "oryza_punctata": 94,
    "asparagus_officinalis": 95,
    "solanum_tuberosum": 96,
    "chondrus_crispus": 97,
    "cucumis_melo": 98,
    "vigna_angularis": 99,
    "sorghum_bicolor": 100,
    "rosa_chinensis": 101,
    "pistacia_vera": 102,
    "lupinus_angustifolius": 103,
    "saccharum_spontaneum": 104,
    "daucus_carota": 105,
    "Nymphaea_colorata": 106,
    "nicotiana_attenuata": 107,
    "cyanidioschyzon_merolae": 108,
    "triticum_aestivum_lancer": 109,
    "manihot_esculenta": 110,
    "eragrostis_curvula": 111,
    "ficus_carica": 112,
    "citrullus_lanatus": 113,
    "glycine_max": 114,
}


class TokenizerCollator:
    def __init__(self, tokenizer, bucket_dir, species, max_len=8192):
        self.tokenizer = tokenizer
        self.species = species
        self.bucket_dir = bucket_dir
        self.max_len = max_len

    def load_from_bucket(self, gene_id):
        md5 = hashlib.md5(gene_id.encode("ascii")).hexdigest()
        bucket = md5[:4]  # Use first 4 MD5 tokens as bucket ID.
        path = os.path.join(f"{self.bucket_dir}/{bucket}.json.gz")
        doc = json.load(gzip.open(path, "rt"))
        return doc[gene_id]

    def __call__(self, examples):
        data = [self.load_from_bucket(ex) for ex in examples]
        tokenized = tokenizer(
            [" ".join(ex["seq"].upper()) for ex in data],
            return_attention_mask=True,
            return_token_type_ids=True,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )

        L = tokenized["input_ids"].shape[-1]
        species_index = torch.tensor([[self.species.get(ex["species"])] for ex in data])
        tokenized["token_type_ids"] = species_index.repeat(1, L)

        tokenized["labels"] = torch.tensor(
            [
                [-100]
                + ex["calls"][: L - 2]
                + [-100]
                + [-100] * (L - 2 - len(ex["calls"]))
                for ex in data
            ]
        )

        return tokenized


class plTrainHarness(lightning.pytorch.LightningModule):
    def __init__(self, model, max_seq_length=468):
        super().__init__()
        self.model = model
        self.max_seq_length = max_seq_length

    def configure_optimizers(self):
        # Linear warmup for 10%, then linear decay for remaining 90%.
        n_steps = self.trainer.estimated_stepping_batches
        n_warmup_steps = int(0.1 * n_steps)
        n_decay_steps = int(0.9 * n_steps)
        optimizer = torch.optim.Adam(self.trainer.model.parameters(), lr=5e-5)
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=n_warmup_steps
        )
        linear_decay = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=0.01, total_iters=n_decay_steps
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup, linear_decay],
            milestones=[n_warmup_steps],
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def training_step(self, batch, batch_idx):
        # BigBird sets attention type to "original_full" if sequences
        # are too short. Always put it back to "block_sparse" to avoid OOM.
        self.model.bert.set_attention_type("block_sparse")
        outputs = self.model(**batch)
        (current_lr,) = self.lr_schedulers().get_last_lr()
        info = {"loss": outputs.loss, "lr": current_lr}
        self.log_dict(dictionary=info, on_step=True, prog_bar=True, logger=True)
        return outputs.loss

    def on_validation_epoch_start(self):
        super().on_validation_epoch_start()
        self.val_output_list = list()

    def validation_step(self, batch, batch_idx):
        self.model.bert.set_attention_type("block_sparse")
        outputs = self.model(**batch)
        self.val_output_list.append(outputs.loss)

    def on_validation_epoch_end(self):
        avg_loss = torch.stack(self.val_output_list).mean()
        self.log(
            "validation_loss", avg_loss, prog_bar=False, logger=True, sync_dist=True
        )


if __name__ == "__main__":
    lightning.pytorch.seed_everything(123)
    torch.set_float32_matmul_precision("medium")

    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file="./TokenizerModel/model.json",
        bos_token="[CLS]",
        eos_token="[SEP]",
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
    )

    # The BigBro model (BigBird with RoFormer position encoding).
    config = transformers.BigBirdConfig(
        vocab_size=len(tokenizer),
        attention_type="block_sparse",
        max_position_embeddings=8192,
        sep_token_id=2,
        type_vocab_size=115,
        # Config options for the RoFormer.
        embedding_size=768,
        rotary_value=False,
    )
    model = bigbro.BigBroForTokenClassification(config=config)

    wrapped_model = bigbro.BigBroForTokenClassification.from_pretrained(
        "./PretrainedBigBro"
    )
    state_dict = wrapped_model.state_dict()
    # Remove the token type embeddings from the pre-trained model
    # because we will recast it to the total number of species.
    state_dict.pop("bert.embeddings.token_type_embeddings.weight")
    keys = model.load_state_dict(state_dict, strict=False)
    assert keys.missing_keys == ["bert.embeddings.token_type_embeddings.weight"]
    assert keys.unexpected_keys == []
    # Delete the wrapped model (the weights are already in the new model).
    del wrapped_model

    train_identifiers = [
        line.rstrip() for line in gzip.open("train_identifiers_115.txt.gz", "rt")
    ]
    val_identifiers = [
        line.rstrip() for line in gzip.open("val_identifiers_115.txt.gz", "rt")
    ]

    harnessed_model = plTrainHarness(model)

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_identifiers,
        collate_fn=TokenizerCollator(tokenizer, "train_buckets_115", species),
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=0 if DEBUG else 4,
        persistent_workers=False if DEBUG else True,
        pin_memory=False if DEBUG else True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_identifiers,
        collate_fn=TokenizerCollator(tokenizer, "val_buckets_115", species),
        batch_size=VALIDATION_BATCH_SIZE,
        num_workers=0 if DEBUG else 4,
        persistent_workers=False if DEBUG else True,
        pin_memory=False if DEBUG else True,
    )

    save_checkpoint = lightning.pytorch.callbacks.ModelCheckpoint(
        dirpath="checkpoints",
        filename="greenbig_checkpoint.pt",
        save_top_k=-1,  # Keep all checkpoints.
        every_n_train_steps=6000,
    )

    trainer = lightning.pytorch.Trainer(
        default_root_dir=".",
        strategy=lightning.pytorch.strategies.DeepSpeedStrategy(stage=2),
        accelerator="gpu",
        devices=1 if DEBUG else -1,
        precision="bf16-mixed",
        max_epochs=MAX_EPOCHS,
        gradient_clip_val=1.0,
        logger=lightning.pytorch.loggers.CSVLogger("."),
        log_every_n_steps=1,
        # Checkpointing.
        enable_checkpointing=True,
        callbacks=[save_checkpoint],
        val_check_interval=12_200,
    )

    ckpt_path = None if len(sys.argv) < 2 else sys.argv[1]
    trainer.validate(harnessed_model, val_dataloader)
    trainer.fit(harnessed_model, train_dataloader, val_dataloader, ckpt_path=ckpt_path)
    trainer.validate(harnessed_model, val_dataloader)

    torch.save(model.state_dict(), "trained_model_all_115.pt")
