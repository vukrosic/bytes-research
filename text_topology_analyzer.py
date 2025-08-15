import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings('ignore')

# Import TDA libraries
try:
    import ripser
    from persim import plot_diagrams
    from ripser import ripser
    TDA_AVAILABLE = True
except ImportError:
    print("Installing required TDA libraries...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'ripser', 'persim', 'scikit-tda'])
    import ripser
    from persim import plot_diagrams
    from ripser import ripser
    TDA_AVAILABLE = True

class TextTopologyAnalyzer:
    """Analyze the topological structure of text at the byte level"""
    
    def __init__(self, text):
        self.text = text
        self.bytes = np.array([ord(c) for c in text])
        self.byte_pairs = self._create_byte_pairs()
        self.byte_triplets = self._create_byte_triplets()
        
    def _create_byte_pairs(self):
        """Create consecutive byte pairs for 2D analysis"""
        if len(self.bytes) < 2:
            return np.array([])
        return np.array([[self.bytes[i], self.bytes[i+1]] for i in range(len(self.bytes)-1)])
    
    def _create_byte_triplets(self):
        """Create consecutive byte triplets for 3D analysis"""
        if len(self.bytes) < 3:
            return np.array([])
        return np.array([[self.bytes[i], self.bytes[i+1], self.bytes[i+2]] 
                        for i in range(len(self.bytes)-2)])
    
    def analyze_byte_distribution(self):
        """Analyze the distribution of bytes in the text"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Individual byte distribution
        axes[0, 0].hist(self.bytes, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Byte Value Distribution')
        axes[0, 0].set_xlabel('Byte Value (ASCII)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Character type analysis
        char_types = {'letters': 0, 'digits': 0, 'spaces': 0, 'punctuation': 0, 'other': 0}
        for byte_val in self.bytes:
            char = chr(byte_val)
            if char.isalpha():
                char_types['letters'] += 1
            elif char.isdigit():
                char_types['digits'] += 1
            elif char.isspace():
                char_types['spaces'] += 1
            elif char in '.,!?;:':
                char_types['punctuation'] += 1
            else:
                char_types['other'] += 1
        
        axes[0, 1].bar(char_types.keys(), char_types.values(), color='green', alpha=0.7)
        axes[0, 1].set_title('Character Type Distribution')
        axes[0, 1].set_ylabel('Count')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45)
        
        # Byte sequence plot
        axes[0, 2].plot(range(len(self.bytes)), self.bytes, 'o-', markersize=2, alpha=0.7)
        axes[0, 2].set_title('Byte Sequence')
        axes[0, 2].set_xlabel('Position in Text')
        axes[0, 2].set_ylabel('Byte Value')
        
        # 2D byte pairs visualization
        if len(self.byte_pairs) > 0:
            axes[1, 0].scatter(self.byte_pairs[:, 0], self.byte_pairs[:, 1], 
                             alpha=0.6, s=20, c=range(len(self.byte_pairs)), cmap='viridis')
            axes[1, 0].set_title('Consecutive Byte Pairs')
            axes[1, 0].set_xlabel('First Byte')
            axes[1, 0].set_ylabel('Second Byte')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Byte transition matrix
        if len(self.byte_pairs) > 0:
            # Create transition matrix for common ASCII range
            ascii_min, ascii_max = 32, 126  # Printable ASCII
            transition_matrix = np.zeros((ascii_max - ascii_min + 1, ascii_max - ascii_min + 1))
            
            for pair in self.byte_pairs:
                if ascii_min <= pair[0] <= ascii_max and ascii_min <= pair[1] <= ascii_max:
                    transition_matrix[pair[0] - ascii_min, pair[1] - ascii_min] += 1
            
            # Only show if we have transitions
            if transition_matrix.sum() > 0:
                im = axes[1, 1].imshow(transition_matrix, cmap='hot', interpolation='nearest')
                axes[1, 1].set_title('Byte Transition Matrix')
                axes[1, 1].set_xlabel('Next Byte (ASCII - 32)')
                axes[1, 1].set_ylabel('Current Byte (ASCII - 32)')
                plt.colorbar(im, ax=axes[1, 1])
        
        # 3D visualization of triplets
        if len(self.byte_triplets) > 0:
            ax = fig.add_subplot(2, 3, 6, projection='3d')
            scatter = ax.scatter(self.byte_triplets[:, 0], self.byte_triplets[:, 1], 
                               self.byte_triplets[:, 2], c=range(len(self.byte_triplets)), 
                               cmap='plasma', s=20, alpha=0.6)
            ax.set_xlabel('Byte 1')
            ax.set_ylabel('Byte 2')
            ax.set_zlabel('Byte 3')
            ax.set_title('Consecutive Byte Triplets')
        
        plt.tight_layout()
        plt.show()
        
        # Print statistics
        print(f"\n--- Text Statistics ---")
        print(f"Text length: {len(self.text)} characters")
        print(f"Unique bytes: {len(np.unique(self.bytes))}")
        print(f"Byte range: [{self.bytes.min()}, {self.bytes.max()}]")
        print(f"Mean byte value: {self.bytes.mean():.2f}")
        print(f"Byte std: {self.bytes.std():.2f}")
        print(f"Character types: {char_types}")
    
    def compute_text_persistent_homology(self):
        """Compute persistent homology of text byte patterns"""
        print("\n--- Computing Persistent Homology of Text ---")
        
        if len(self.byte_pairs) == 0:
            print("Not enough data for persistent homology analysis")
            return None
        
        # Normalize byte pairs to [0, 1] range
        normalized_pairs = self.byte_pairs / 255.0
        
        # Compute persistence diagrams
        print(f"Computing persistence on {len(normalized_pairs)} byte pairs...")
        dgms = ripser(normalized_pairs, maxdim=2)['dgms']
        
        # Plot persistence diagrams
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, dgm in enumerate(dgms):
            if i >= 3:
                break
            
            if len(dgm) > 0:
                # Filter out infinite points for plotting
                finite_dgm = dgm[dgm[:, 1] < np.inf]
                if len(finite_dgm) > 0:
                    axes[i].scatter(finite_dgm[:, 0], finite_dgm[:, 1], s=30, alpha=0.6)
                    max_death = finite_dgm[:, 1].max()
                    axes[i].plot([0, max_death], [0, max_death], 'k--', alpha=0.3)
                    
                    # Highlight most persistent features
                    if len(finite_dgm) > 0:
                        persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
                        most_persistent_idx = np.argmax(persistences)
                        axes[i].scatter(finite_dgm[most_persistent_idx, 0], 
                                      finite_dgm[most_persistent_idx, 1], 
                                      s=100, c='red', marker='*', 
                                      label=f'Most persistent: {persistences[most_persistent_idx]:.4f}')
                        axes[i].legend()
                else:
                    axes[i].text(0.5, 0.5, 'Only infinite features', ha='center', va='center', 
                               transform=axes[i].transAxes)
            else:
                axes[i].text(0.5, 0.5, 'No features', ha='center', va='center', 
                           transform=axes[i].transAxes)
            
            axes[i].set_xlabel('Birth')
            axes[i].set_ylabel('Death')
            axes[i].set_title(f'H_{i} Persistence Diagram')
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Persistent Homology of Text Byte Patterns')
        plt.tight_layout()
        plt.show()
        
        # Analyze topological features
        self._analyze_text_topology_features(dgms)
        
        return dgms
    
    def _analyze_text_topology_features(self, dgms):
        """Analyze topological features specific to text"""
        print("\n--- Text Topological Features ---")
        
        for i, dgm in enumerate(dgms):
            if i >= 3:
                break
            
            finite_dgm = dgm[dgm[:, 1] < np.inf]
            
            if len(finite_dgm) > 0:
                persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
                print(f"\nH_{i} (Dimension {i}):")
                print(f"  Number of topological features: {len(finite_dgm)}")
                print(f"  Max persistence: {persistences.max():.4f}")
                print(f"  Mean persistence: {persistences.mean():.4f}")
                
                if i == 0:
                    print(f"  → Connected components in byte space")
                elif i == 1:
                    print(f"  → Loops/cycles in byte transition patterns")
                elif i == 2:
                    print(f"  → Voids/cavities in byte space")
                
                # Find most persistent features
                top_indices = np.argsort(persistences)[-3:][::-1]
                print(f"  Most persistent features:")
                for idx in top_indices[:min(3, len(top_indices))]:
                    print(f"    Birth: {finite_dgm[idx, 0]:.4f}, "
                          f"Death: {finite_dgm[idx, 1]:.4f}, "
                          f"Persistence: {persistences[idx]:.4f}")
            else:
                print(f"\nH_{i} (Dimension {i}): No finite features")
    
    def analyze_text_manifold_structure(self):
        """Analyze the manifold structure of text in byte space"""
        print("\n--- Text Manifold Analysis ---")
        
        if len(self.byte_pairs) == 0:
            print("Not enough data for manifold analysis")
            return
        
        # Normalize data
        normalized_pairs = self.byte_pairs / 255.0
        
        # PCA analysis
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(normalized_pairs)
        
        # t-SNE analysis (sample if too large)
        sample_size = min(1000, len(normalized_pairs))
        if len(normalized_pairs) > sample_size:
            indices = np.random.choice(len(normalized_pairs), sample_size, replace=False)
            tsne_data = normalized_pairs[indices]
        else:
            tsne_data = normalized_pairs
            indices = np.arange(len(normalized_pairs))
        
        print("Computing t-SNE embedding...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(tsne_data)//4))
        tsne_result = tsne.fit_transform(tsne_data)
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Original byte pair space
        scatter1 = axes[0].scatter(normalized_pairs[:, 0], normalized_pairs[:, 1], 
                                 c=range(len(normalized_pairs)), cmap='viridis', 
                                 s=20, alpha=0.6)
        axes[0].set_title('Original Byte Pair Space')
        axes[0].set_xlabel('First Byte (normalized)')
        axes[0].set_ylabel('Second Byte (normalized)')
        axes[0].grid(True, alpha=0.3)
        plt.colorbar(scatter1, ax=axes[0], label='Position in Text')
        
        # PCA projection
        scatter2 = axes[1].scatter(pca_result[:, 0], pca_result[:, 1], 
                                 c=range(len(pca_result)), cmap='viridis', 
                                 s=20, alpha=0.6)
        axes[1].set_title(f'PCA Projection (explained var: {pca.explained_variance_ratio_.sum():.2%})')
        axes[1].set_xlabel('PC1')
        axes[1].set_ylabel('PC2')
        axes[1].grid(True, alpha=0.3)
        plt.colorbar(scatter2, ax=axes[1], label='Position in Text')
        
        # t-SNE projection
        scatter3 = axes[2].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                                 c=indices, cmap='viridis', s=20, alpha=0.6)
        axes[2].set_title('t-SNE Projection')
        axes[2].set_xlabel('t-SNE 1')
        axes[2].set_ylabel('t-SNE 2')
        axes[2].grid(True, alpha=0.3)
        plt.colorbar(scatter3, ax=axes[2], label='Position in Text')
        
        plt.suptitle('Manifold Structure of Text in Byte Space')
        plt.tight_layout()
        plt.show()
        
        print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
        print(f"Total variance explained: {pca.explained_variance_ratio_.sum():.2%}")
    
    def analyze_linguistic_patterns(self):
        """Analyze linguistic patterns in the topological structure"""
        print("\n--- Linguistic Pattern Analysis ---")
        
        # Word boundary analysis
        words = self.text.split()
        word_lengths = [len(word) for word in words]
        
        # Sentence analysis
        sentences = [s.strip() for s in self.text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        sentence_lengths = [len(sentence) for sentence in sentences]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Word length distribution
        axes[0, 0].hist(word_lengths, bins=20, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 0].set_title('Word Length Distribution')
        axes[0, 0].set_xlabel('Word Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        
        # Sentence length distribution
        if sentence_lengths:
            axes[0, 1].hist(sentence_lengths, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[0, 1].set_title('Sentence Length Distribution')
            axes[0, 1].set_xlabel('Sentence Length (characters)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Byte value evolution
        axes[1, 0].plot(range(len(self.bytes)), self.bytes, alpha=0.7, linewidth=1)
        axes[1, 0].set_title('Byte Value Evolution')
        axes[1, 0].set_xlabel('Position in Text')
        axes[1, 0].set_ylabel('Byte Value')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Local byte variance (smoothness)
        if len(self.bytes) > 10:
            window_size = min(10, len(self.bytes) // 4)
            local_variance = []
            for i in range(len(self.bytes) - window_size + 1):
                window = self.bytes[i:i + window_size]
                local_variance.append(np.var(window))
            
            axes[1, 1].plot(range(len(local_variance)), local_variance, alpha=0.7, color='red')
            axes[1, 1].set_title(f'Local Byte Variance (window={window_size})')
            axes[1, 1].set_xlabel('Position in Text')
            axes[1, 1].set_ylabel('Local Variance')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        print(f"Number of words: {len(words)}")
        print(f"Average word length: {np.mean(word_lengths):.2f}")
        print(f"Number of sentences: {len(sentences)}")
        if sentence_lengths:
            print(f"Average sentence length: {np.mean(sentence_lengths):.2f}")
        print(f"Text complexity (byte std): {self.bytes.std():.2f}")


def create_sample_text():
    """Create a sample text for analysis"""
    return """
    The quick brown fox jumps over the lazy dog. This pangram contains every letter 
    of the English alphabet at least once, making it perfect for testing typography 
    and analyzing character distributions in topological data analysis.
    
    Mathematics reveals hidden patterns in seemingly random data. When we examine text 
    at the byte level, we discover fascinating geometric structures that emerge from 
    the sequential nature of language. Each character becomes a point in high-dimensional 
    space, and the relationships between consecutive characters form complex manifolds.
    
    Persistent homology allows us to study these structures across multiple scales, 
    revealing how linguistic patterns create topological features. Loops might represent 
    recurring word patterns, while connected components could indicate different 
    semantic clusters in the text.
    
    Numbers like 123, 456, and 789 create different byte patterns than letters. 
    Special characters !@#$%^&*() add even more complexity to the topological landscape.
    """


def main():
    print("=" * 70)
    print("TOPOLOGICAL ANALYSIS OF TEXT BYTES")
    print("=" * 70)
    
    # Create sample text
    sample_text = create_sample_text()
    
    # Initialize analyzer
    analyzer = TextTopologyAnalyzer(sample_text)
    
    print(f"Analyzing text with {len(sample_text)} characters...")
    print(f"First 100 characters: {repr(sample_text[:100])}")
    
    # Analyze byte distribution
    print("\n1. Analyzing byte distribution...")
    analyzer.analyze_byte_distribution()
    
    # Compute persistent homology
    print("\n2. Computing persistent homology...")
    dgms = analyzer.compute_text_persistent_homology()
    
    # Analyze manifold structure
    print("\n3. Analyzing manifold structure...")
    analyzer.analyze_text_manifold_structure()
    
    # Analyze linguistic patterns
    print("\n4. Analyzing linguistic patterns...")
    analyzer.analyze_linguistic_patterns()
    
    # Summary insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS FROM TEXT TOPOLOGY ANALYSIS")
    print("=" * 70)
    print("""
1. BYTE SPACE GEOMETRY:
   - Text creates sparse patterns in 256-dimensional byte space
   - Consecutive characters form trajectories through this space
   - Language constraints create non-uniform distributions

2. TOPOLOGICAL FEATURES:
   - Connected components reveal character clustering
   - Loops indicate recurring linguistic patterns
   - Persistence measures structural stability

3. LINGUISTIC STRUCTURE:
   - Word boundaries create discontinuities in byte space
   - Punctuation marks act as topological landmarks
   - Sentence structure influences manifold geometry

4. APPLICATIONS:
   - Text classification through topological signatures
   - Language detection via persistent homology
   - Compression algorithms based on topological features
   - Anomaly detection in text streams
""")


if __name__ == "__main__":
    main()