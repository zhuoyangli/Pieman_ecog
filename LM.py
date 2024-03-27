import numpy as np
import scipy.stats as ss
import torch
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoTokenizer

INIT = ["I", "We", "She", "He", "They", "It", " I"]


class GPT2:
    """wrapper for https://huggingface.co/openai-gpt"""

    def __init__(
        self,
        path,
        tokenizer=AutoTokenizer.from_pretrained("gpt2"),
        vocab_ids=None,
        device="cpu",
    ):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained("gpt2").eval().to(self.device)
        self.tokenizer = tokenizer
        self.vocab_ids = vocab_ids
        np.random.seed(42)

    def encode(self, words):
        """map from words to ids"""
        return self.tokenizer.encode(words)

    def encode_word_list(self, words):
        return [self.encode(word)[0] for word in words]

    def decode(self, ids):
        """map from ids to words"""
        return self.tokenizer.decode(ids)

    def get_story_array(self, words, context_words):
        """get word ids for each phrase in a stimulus story"""
        nctx = context_words + 1
        story_ids = self.encode(words)
        story_array = np.zeros([len(story_ids), nctx])  # + self.UNK_ID
        for i in range(len(story_array)):
            segment = story_ids[i : i + nctx]
            story_array[i, : len(segment)] = segment
        return torch.tensor(story_array).long()

    def get_context_array(self, contexts):
        """get word ids for each context"""
        context_array = np.array([self.encode(words) for words in contexts])
        return torch.tensor(context_array).long()

    def generate_context_array_from_id(self, ids, n_context_words):
        """generate context array from ids given number of context words"""
        context_array = np.array(
            [ids[i - n_context_words : i] for i in range(n_context_words, len(ids))]
        )
        return torch.tensor(context_array).long()

    def generate_embedding_array_from_id(self, ids, n_context_words):
        """generate embedding array from ids given number of context words"""
        embedding_array = np.array(
            [
                np.pad(
                    ids[: i + 1], pad_width=(n_context_words - i, 0), constant_values=0
                )
                if i < n_context_words
                else ids[i - n_context_words : i + 1]
                for i in range(len(ids))
            ]
        )
        return torch.tensor(embedding_array).long()

    def get_hidden(self, ids, mask, layer):
        """get hidden layer representations"""
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids.to(self.device),
                attention_mask=mask.to(self.device),
                output_hidden_states=True,
            )
        return outputs.hidden_states[layer].detach().cpu().numpy()

    def get_probs(self, ids, mask=None):
        """get next word probability distributions"""
        ids = torch.tensor(ids).to(self.device)
        mask = torch.ones(ids.shape).int()
        with torch.no_grad():
            outputs = self.model(
                input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
            )
        # probs = softmax(outputs.logits, dim = 2).detach().cpu().numpy()
        probs = softmax(outputs.logits, axis=-1)
        return probs

    def get_prob_sorted_id_in_vocab(self, ids, mask):
        """get ids in descending probability order"""
        probs = self.get_probs(ids, mask)
        sorted_id = sorted(self.vocab_ids, key=lambda x: probs[-1][x], reverse=True)
        return sorted_id

    def get_embeddings(self, tokens, n_context_words, layer, pca_dim):
        token_ids = [self.tokenizer.encode(token)[0] for token in tokens]
        embedding_arrays = self.generate_embedding_array_from_id(
            token_ids, n_context_words
        )
        attention_mask = embedding_arrays > 0
        xtmp = self.get_hidden(embedding_arrays, attention_mask, layer)[:, -1, :]

        # standardize embedding dimensions
        scaler = StandardScaler().fit(
            xtmp
        )  # assumes shape = (n_samples, n_features), scales along the features axis
        xtmp = scaler.transform(xtmp)
        means = scaler.mean_
        scales = scaler.scale_

        # do dimensionality reduction here
        pca = PCA(n_components=pca_dim, whiten=True).fit(xtmp)
        pca_weights = pca.components_.T * (pca.explained_variance_**-0.5)
        xtmp = xtmp @ pca_weights
        return xtmp, means, scales, pca_weights


def get_nucleus(probs, nuc_mass, nuc_ratio):
    """identify words that constitute a given fraction of the probability mass"""
    nuc_ids = np.where(probs[-1] >= np.max(probs[-1]) * nuc_ratio)[0]
    nuc_pairs = sorted(zip(nuc_ids, probs[-1][nuc_ids]), key=lambda x: -x[1])
    sum_mass = np.cumsum([x[1] for x in nuc_pairs])
    cutoffs = np.where(sum_mass >= nuc_mass)[0]
    if len(cutoffs) > 0:
        nuc_pairs = nuc_pairs[: cutoffs[0] + 1]
    nuc_ids = [x[0] for x in nuc_pairs]
    return nuc_ids


class Decoder:
    """class for generating word sequences from language model"""

    def __init__(
        self, model, ids, nuc_mass=1.0, nuc_ratio=0.0, beam_width=100, extensions=10
    ):
        self.model = model
        self.ids = ids
        self.nuc_mass, self.nuc_ratio = nuc_mass, nuc_ratio
        self.beam_width, self.extensions = beam_width, extensions
        self.beam = [Hypothesis()]  # initialize with empty hypothesis
        self.scored_extensions = []  # global extension pool

    def ps(self, contexts):
        """generate next word probability distributions"""
        ids = self.model.get_context_array(contexts)
        mask = torch.ones(ids.shape).int()
        probs = self.model.get_probs(ids, mask)
        return probs[:, len(contexts[0]) - 1]

    def beam_propose(self, beam, context_words):
        """get possible extensions of beam"""
        if len(beam) == 1:
            nuc_ids = [
                self.model.encode(w)[0]
                for w in INIT
                if self.model.encode(w)[0] in self.ids
            ]
            nuc_logprobs = np.log(np.ones(len(nuc_ids)) / len(nuc_ids))
            return [(nuc_ids, nuc_logprobs)]
        else:
            contexts = [hyp.ids[-context_words:] for hyp in beam]
            beam_probs = self.model.get_probs(contexts)
            beam_nucs = []
            for context, probs in zip(contexts, beam_probs):
                nuc_ids = get_nucleus(
                    probs, nuc_mass=self.nuc_mass, nuc_ratio=self.nuc_ratio
                )
                nuc_ids = [i for i in nuc_ids if i in self.ids]
                nuc_logprobs = np.log([probs[-1][id] for id in nuc_ids])
                beam_nucs.append((nuc_ids, nuc_logprobs))
            return beam_nucs

    def get_hypotheses(self):
        """get the number of permitted extensions for each hypothesis on the beam"""
        if len(self.beam[0].ids) == 0:
            return zip(self.beam, [self.extensions for hypothesis in self.beam])
        logprobs = [sum(hypothesis.logprobs) for hypothesis in self.beam]
        num_extensions = [
            int(np.ceil(self.extensions * rank / len(logprobs)))
            for rank in ss.rankdata(logprobs)
        ]
        return zip(self.beam, num_extensions)

    def add_extensions(self, extensions, likelihoods, num_extensions):
        """add extensions for each hypothesis to global extension pool"""
        scored_extensions = sorted(zip(extensions, likelihoods), key=lambda x: -x[1])
        self.scored_extensions.extend(scored_extensions[:num_extensions])

    def extend(self, verbose=False):
        """update beam based on global extension pool"""
        self.beam = [
            x[0]
            for x in sorted(self.scored_extensions, key=lambda x: -x[1])[
                : self.beam_width
            ]
        ]
        self.scored_extensions = []
        if verbose:
            print(self.model.decode(self.beam[0].ids))

    def save(self, path):
        """save decoder results"""
        np.savez(path, ids=np.array(self.beam[0].ids))


class Hypothesis(object):
    """a class for representing word sequence hypotheses"""

    def __init__(self, parent=None, extension=None):
        if parent is None:
            self.ids, self.logprobs, self.embs = [], [], []
        else:
            id, logprob, emb = extension
            self.ids = parent.ids + [id]
            self.logprobs = parent.logprobs + [logprob]
            self.embs = parent.embs + [emb]
