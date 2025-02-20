class BPE:
    def __init__(self,train_input, vocab_size=276):
        self.input = train_input
        self.vocab_size = vocab_size
        self.vocab = {idx: bytes([idx]) for idx in range(256)}
        self.train()
        
    def train(self): 
        self.tokens = self.input.encode("utf-8")
        self.tokens = list(map(int,self.tokens))
        num_merge = self.vocab_size - 256
        ids = list(self.tokens)
        self.merges = {}
        for i in range(num_merge):
            stats = self.get_stat(ids) 
            if not stats:
                break
            pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f'Merging {pair} into a new token {idx}')
            ids = self.merge(ids,pair,idx)
            self.merges[pair] = idx
        # update vocab so that it can be used for later
        for (p0,p1),idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
        
    
    def get_stat(self,ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair,0) + 1
        return counts
    
    def merge(self,ids,pair,idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i+=1
        return newids
    def decode(self,ids: list[int]) -> str:
        # given ids, return python string
        tokens = b"".join(self.vocab[idx] for idx in ids)
        return tokens.decode("utf-8","replace")
    
    def encode(self,text) -> list[int]:
        tokens = list(text.encode("utf-8"))
        while len(tokens) >= 2:
            stats = self.get_stat(tokens)
            pair = min(stats, key= lambda p: self.merges.get(p,float('inf')))
            if pair not in self.merges: # no more merges to be made
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens,pair,idx)
        return tokens
            
        
                
if __name__ == '__main__':
    train_input = '''Artificial intelligence (AI) is transforming the world at an unprecedented pace. From self-driving cars to medical diagnostics, AI systems are revolutionizing industries and changing the way humans live and work. Machine learning, a subset of AI, allows systems to learn from data and improve their performance over time without being explicitly programmed. Deep learning, a specialized form of machine learning, uses neural networks with multiple layers to process complex data such as images, audio, and natural language.

In recent years, advances in AI have led to remarkable breakthroughs in various fields. Natural language processing (NLP) enables computers to understand, interpret, and generate human language, facilitating applications such as chatbots, virtual assistants, and automated translation services. In healthcare, AI-powered algorithms are being used to detect diseases, develop treatment plans, and even predict patient outcomes. In finance, AI is employed to detect fraudulent transactions, automate trading, and provide personalized financial advice.

Despite its many benefits, AI also raises significant ethical and societal concerns. Issues such as data privacy, algorithmic bias, and job displacement are critical challenges that must be addressed to ensure the responsible and fair deployment of AI technologies. Policymakers, researchers, and industry leaders must work together to establish guidelines and best practices for the development and use of AI systems.

The future of AI holds immense potential. As research continues to advance, AI systems are expected to become even more powerful and versatile. Emerging technologies such as quantum computing may further accelerate AI innovation, opening up new possibilities and applications. However, with great power comes great responsibility. Ensuring that AI is developed and used in a manner that benefits society as a whole is a task that will require ongoing collaboration, vigilance, and thoughtful consideration.

Whether we are ready or not, the AI-driven future is rapidly approaching. It is up to all of us to navigate this new era with wisdom and foresight, leveraging the power of AI to create a better, more inclusive, and more sustainable world.
'''
    bpe = BPE(train_input)