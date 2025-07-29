import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import re
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VendorMetrics:
    """Vendor performance metrics"""
    channel: str
    total_posts: int
    avg_views_per_post: float
    posting_frequency: float  # posts per week
    avg_price: float
    top_performing_post: Dict
    lending_score: float
    engagement_rate: float
    price_range: Tuple[float, float]
    product_diversity: int

class VendorAnalyticsEngine:
    """Analytics engine for vendor performance and lending assessment"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.df = None
        self.vendor_metrics = {}
        self._load_data()
        
    def _load_data(self):
        """Load and preprocess vendor data"""
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} records from {self.data_path}")
            
            # Convert timestamp to datetime
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
            
            # Extract price information from text
            self.df['extracted_price'] = self.df['text'].apply(self._extract_price)
            
            # Extract product information
            self.df['extracted_products'] = self.df['text'].apply(self._extract_products)
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _extract_price(self, text: str) -> Optional[float]:
        """Extract price from text using regex patterns"""
        if not isinstance(text, str):
            return None
        
        # Price patterns for Amharic text
        price_patterns = [
            r'ዋጋ\s*[:፡]?\s*(\d+(?:,\d{3})*)\s*(?:ብር|ETB|birr|Br)',
            r'(\d+(?:,\d{3})*)\s*(?:ብር|ETB|birr|Br)',
            r'price\s*[:፡]?\s*(\d+(?:,\d{3})*)',
            r'(\d+(?:,\d{3})*)\s*ETB'
        ]
        
        for pattern in price_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                price_str = match.group(1).replace(',', '')
                try:
                    return float(price_str)
                except ValueError:
                    continue
        
        return None
    
    def _extract_products(self, text: str) -> List[str]:
        """Extract product names from text"""
        if not isinstance(text, str):
            return []
        
        # Product keywords and patterns
        product_patterns = [
            r'\b(LCD|Tablet|ስልክ|ኮምፒዩተር|ቴሌቪዥን|ላፕቶፕ|ስልክ|ስማርትፎን)\b',
            r'\b([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\b',
            r'(\d+\.?\d*\s*(?:inch|cm|ሜትር))\b'
        ]
        
        products = []
        for pattern in product_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            products.extend(matches)
        
        return list(set(products))  # Remove duplicates
    
    def calculate_vendor_metrics(self, channel: str) -> VendorMetrics:
        """Calculate comprehensive metrics for a vendor channel"""
        logger.info(f"Calculating metrics for {channel}")
        
        # Filter data for the channel
        channel_data = self.df[self.df['channel'] == channel].copy()
        
        if len(channel_data) == 0:
            logger.warning(f"No data found for channel {channel}")
            return None
        
        # Basic metrics
        total_posts = len(channel_data)
        avg_views_per_post = channel_data['views'].mean()
        
        # Posting frequency (posts per week)
        if len(channel_data) > 1:
            date_range = channel_data['timestamp'].max() - channel_data['timestamp'].min()
            weeks = max(1, date_range.days / 7)
            posting_frequency = total_posts / weeks
        else:
            posting_frequency = 0
        
        # Price analysis
        prices = channel_data['extracted_price'].dropna()
        avg_price = prices.mean() if len(prices) > 0 else 0
        price_range = (prices.min(), prices.max()) if len(prices) > 0 else (0, 0)
        
        # Top performing post
        top_post_idx = channel_data['views'].idxmax()
        top_post = {
            'message_id': channel_data.loc[top_post_idx, 'message_id'],
            'views': channel_data.loc[top_post_idx, 'views'],
            'text': channel_data.loc[top_post_idx, 'text'][:100] + "...",
            'price': channel_data.loc[top_post_idx, 'extracted_price'],
            'products': channel_data.loc[top_post_idx, 'extracted_products']
        }
        
        # Product diversity
        all_products = []
        for products in channel_data['extracted_products']:
            all_products.extend(products)
        product_diversity = len(set(all_products))
        
        # Engagement rate (views per post normalized by time)
        engagement_rate = avg_views_per_post / max(1, posting_frequency)
        
        # Calculate lending score
        lending_score = self._calculate_lending_score(
            avg_views_per_post, posting_frequency, avg_price, engagement_rate
        )
        
        return VendorMetrics(
            channel=channel,
            total_posts=total_posts,
            avg_views_per_post=avg_views_per_post,
            posting_frequency=posting_frequency,
            avg_price=avg_price,
            top_performing_post=top_post,
            lending_score=lending_score,
            engagement_rate=engagement_rate,
            price_range=price_range,
            product_diversity=product_diversity
        )
    
    def _calculate_lending_score(self, avg_views: float, posting_freq: float, 
                               avg_price: float, engagement_rate: float) -> float:
        """Calculate lending score based on multiple factors"""
        
        # Normalize each metric to 0-1 scale
        views_score = min(avg_views / 1000, 1.0)  # Cap at 1000 views
        frequency_score = min(posting_freq / 10, 1.0)  # Cap at 10 posts/week
        price_score = min(avg_price / 10000, 1.0)  # Cap at 10,000 ETB
        engagement_score = min(engagement_rate / 100, 1.0)  # Cap at 100 engagement rate
        
        # Weighted combination
        lending_score = (
            views_score * 0.3 +
            frequency_score * 0.25 +
            price_score * 0.2 +
            engagement_score * 0.25
        )
        
        return round(lending_score * 100, 2)  # Convert to percentage
    
    def analyze_all_vendors(self) -> Dict[str, VendorMetrics]:
        """Analyze all vendor channels"""
        logger.info("Analyzing all vendors...")
        
        channels = self.df['channel'].unique()
        
        for channel in channels:
            try:
                metrics = self.calculate_vendor_metrics(channel)
                if metrics:
                    self.vendor_metrics[channel] = metrics
                    logger.info(f"Completed analysis for {channel}")
            except Exception as e:
                logger.error(f"Error analyzing {channel}: {e}")
        
        return self.vendor_metrics
    
    def generate_vendor_scorecard(self, output_path: str = "results/vendor_scorecard.csv"):
        """Generate vendor scorecard report"""
        logger.info("Generating vendor scorecard...")
        
        if not self.vendor_metrics:
            self.analyze_all_vendors()
        
        # Create scorecard DataFrame
        scorecard_data = []
        
        for channel, metrics in self.vendor_metrics.items():
            scorecard_data.append({
                'Channel': channel,
                'Total Posts': metrics.total_posts,
                'Avg Views/Post': round(metrics.avg_views_per_post, 2),
                'Posts/Week': round(metrics.posting_frequency, 2),
                'Avg Price (ETB)': round(metrics.avg_price, 2),
                'Lending Score (%)': metrics.lending_score,
                'Engagement Rate': round(metrics.engagement_rate, 2),
                'Product Diversity': metrics.product_diversity,
                'Price Range (ETB)': f"{metrics.price_range[0]:.0f} - {metrics.price_range[1]:.0f}"
            })
        
        scorecard_df = pd.DataFrame(scorecard_data)
        
        # Sort by lending score
        scorecard_df = scorecard_df.sort_values('Lending Score (%)', ascending=False)
        
        # Save to CSV
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        scorecard_df.to_csv(output_path, index=False, encoding='utf-8')
        
        logger.info(f"Vendor scorecard saved to {output_path}")
        return scorecard_df
    
    def create_visualizations(self, output_dir: str = "results"):
        """Create visualizations for vendor analysis"""
        logger.info("Creating visualizations...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        if not self.vendor_metrics:
            self.analyze_all_vendors()
        
        # Prepare data for visualization
        channels = list(self.vendor_metrics.keys())
        lending_scores = [metrics.lending_score for metrics in self.vendor_metrics.values()]
        avg_views = [metrics.avg_views_per_post for metrics in self.vendor_metrics.values()]
        posting_freq = [metrics.posting_frequency for metrics in self.vendor_metrics.values()]
        avg_prices = [metrics.avg_price for metrics in self.vendor_metrics.values()]
        
        # 1. Lending Score Comparison
        plt.figure(figsize=(12, 6))
        bars = plt.bar(channels, lending_scores, color='skyblue')
        plt.title('Vendor Lending Scores', fontsize=14, fontweight='bold')
        plt.xlabel('Vendor Channels')
        plt.ylabel('Lending Score (%)')
        plt.xticks(rotation=45)
        plt.ylim(0, 100)
        
        # Add value labels on bars
        for bar, score in zip(bars, lending_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{score:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/lending_scores.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Performance Metrics Heatmap
        metrics_data = {
            'Lending Score': lending_scores,
            'Avg Views': avg_views,
            'Posting Frequency': posting_freq,
            'Avg Price': avg_prices
        }
        
        metrics_df = pd.DataFrame(metrics_data, index=channels)
        
        # Normalize for heatmap
        metrics_normalized = (metrics_df - metrics_df.min()) / (metrics_df.max() - metrics_df.min())
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(metrics_normalized.T, annot=True, cmap='YlOrRd', fmt='.2f')
        plt.title('Vendor Performance Metrics (Normalized)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/performance_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Scatter plot: Views vs Lending Score
        plt.figure(figsize=(10, 6))
        plt.scatter(avg_views, lending_scores, s=100, alpha=0.7)
        
        for i, channel in enumerate(channels):
            plt.annotate(channel, (avg_views[i], lending_scores[i]), 
                        xytext=(5, 5), textcoords='offset points')
        
        plt.xlabel('Average Views per Post')
        plt.ylabel('Lending Score (%)')
        plt.title('Views vs Lending Score Correlation')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/views_vs_score.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def generate_lending_recommendations(self, output_path: str = "results/lending_recommendations.json"):
        """Generate lending recommendations based on vendor analysis"""
        logger.info("Generating lending recommendations...")
        
        if not self.vendor_metrics:
            self.analyze_all_vendors()
        
        recommendations = {
            'high_priority': [],
            'medium_priority': [],
            'low_priority': [],
            'summary': {}
        }
        
        for channel, metrics in self.vendor_metrics.items():
            vendor_rec = {
                'channel': channel,
                'lending_score': metrics.lending_score,
                'strengths': [],
                'concerns': [],
                'recommended_loan_amount': 0,
                'risk_level': 'low'
            }
            
            # Analyze strengths and concerns
            if metrics.avg_views_per_post > 500:
                vendor_rec['strengths'].append('High engagement')
            if metrics.posting_frequency > 5:
                vendor_rec['strengths'].append('Consistent activity')
            if metrics.avg_price > 5000:
                vendor_rec['strengths'].append('High-value products')
            if metrics.product_diversity > 5:
                vendor_rec['strengths'].append('Product diversity')
            
            if metrics.avg_views_per_post < 100:
                vendor_rec['concerns'].append('Low engagement')
            if metrics.posting_frequency < 2:
                vendor_rec['concerns'].append('Inconsistent activity')
            if metrics.avg_price < 1000:
                vendor_rec['concerns'].append('Low-value products')
            
            # Determine risk level
            if metrics.lending_score >= 70:
                vendor_rec['risk_level'] = 'low'
                vendor_rec['recommended_loan_amount'] = int(metrics.avg_price * 10)
            elif metrics.lending_score >= 50:
                vendor_rec['risk_level'] = 'medium'
                vendor_rec['recommended_loan_amount'] = int(metrics.avg_price * 5)
            else:
                vendor_rec['risk_level'] = 'high'
                vendor_rec['recommended_loan_amount'] = int(metrics.avg_price * 2)
            
            # Categorize by priority
            if metrics.lending_score >= 70:
                recommendations['high_priority'].append(vendor_rec)
            elif metrics.lending_score >= 50:
                recommendations['medium_priority'].append(vendor_rec)
            else:
                recommendations['low_priority'].append(vendor_rec)
        
        # Summary statistics
        recommendations['summary'] = {
            'total_vendors': len(self.vendor_metrics),
            'high_priority_count': len(recommendations['high_priority']),
            'medium_priority_count': len(recommendations['medium_priority']),
            'low_priority_count': len(recommendations['low_priority']),
            'avg_lending_score': np.mean([m.lending_score for m in self.vendor_metrics.values()]),
            'total_recommended_loan_amount': sum(
                rec['recommended_loan_amount'] for rec in 
                recommendations['high_priority'] + recommendations['medium_priority']
            )
        }
        
        # Save recommendations
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(recommendations, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Lending recommendations saved to {output_path}")
        return recommendations

def main():
    """Main function to run vendor analytics"""
    # Example usage
    data_path = "data/processed/telegram_processed.csv"
    
    try:
        # Initialize analytics engine
        engine = VendorAnalyticsEngine(data_path)
        
        # Generate scorecard
        scorecard = engine.generate_vendor_scorecard()
        print("Vendor Scorecard:")
        print(scorecard.to_string(index=False))
        
        # Create visualizations
        engine.create_visualizations()
        
        # Generate lending recommendations
        recommendations = engine.generate_lending_recommendations()
        
        print("\nLending Recommendations Summary:")
        print(f"Total vendors analyzed: {recommendations['summary']['total_vendors']}")
        print(f"High priority vendors: {recommendations['summary']['high_priority_count']}")
        print(f"Medium priority vendors: {recommendations['summary']['medium_priority_count']}")
        print(f"Low priority vendors: {recommendations['summary']['low_priority_count']}")
        print(f"Average lending score: {recommendations['summary']['avg_lending_score']:.2f}%")
        print(f"Total recommended loan amount: {recommendations['summary']['total_recommended_loan_amount']:,} ETB")
        
    except Exception as e:
        logger.error(f"Error in vendor analytics: {e}")

if __name__ == "__main__":
    main() 