"""
COMPLETE DATA INTEGRATION FOR UPSCALED COLLECTORS
Merges all collected datasets into unified nodes and edges
Target: 113,600+ rows → Unified network for Neo4j
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import logging
import glob

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class UpscaledDataIntegrator:
    """
    Integrates all upscaled datasets into unified multi-layer network
    """
    
    def __init__(self, data_root='data'):
        self.data_root = Path(data_root)
        self.output_dir = self.data_root / 'integrated'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.datasets = {}
        
    def load_all_upscaled_datasets(self):
        """
        Step 1: Load all upscaled datasets
        Time: 30 minutes
        """
        logger.info("📂 Loading all upscaled datasets...")
        
        # 1. BANKING UPSCALED
        banking_upscaled = self.data_root / 'banking_upscaled'
        if banking_upscaled.exists():
            logger.info("\n🏦 Loading Banking Upscaled...")
            
            # Banking indicators timeseries
            files = list(banking_upscaled.glob('banking_indicators_timeseries_*.csv'))
            if files:
                self.datasets['banking_indicators'] = pd.read_csv(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"  ✓ Banking indicators: {len(self.datasets['banking_indicators']):,} rows")
            
            # Interbank lending network
            files = list(banking_upscaled.glob('interbank_lending_network_*.csv'))
            if files:
                self.datasets['interbank_lending'] = pd.read_csv(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"  ✓ Interbank lending: {len(self.datasets['interbank_lending']):,} rows")
            
            # Institution profiles
            files = list(banking_upscaled.glob('institution_profiles_*.csv'))
            if files:
                self.datasets['institution_profiles'] = pd.read_csv(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"  ✓ Institution profiles: {len(self.datasets['institution_profiles']):,} rows")
            
            # Stress events
            files = list(banking_upscaled.glob('stress_events_*.csv'))
            if files:
                self.datasets['stress_events'] = pd.read_csv(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"  ✓ Stress events: {len(self.datasets['stress_events']):,} rows")
            
            # Daily positions
            files = list(banking_upscaled.glob('daily_positions_*.csv'))
            if files:
                self.datasets['daily_positions'] = pd.read_csv(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"  ✓ Daily positions: {len(self.datasets['daily_positions']):,} rows")
        
        # 2. MARKET UPSCALED
        market_upscaled = self.data_root / 'market_upscaled'
        if market_upscaled.exists():
            logger.info("\n📈 Loading Market Upscaled...")
            
            # Prices with indicators
            files = list(market_upscaled.glob('prices_with_indicators_*.csv'))
            if files:
                self.datasets['market_prices'] = pd.read_csv(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"  ✓ Market prices: {len(self.datasets['market_prices']):,} rows")
            
            # Market correlations
            files = list(market_upscaled.glob('market_correlations_*.csv'))
            if files:
                self.datasets['market_correlations'] = pd.read_csv(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"  ✓ Market correlations: {len(self.datasets['market_correlations']):,} rows")
            
            # Company fundamentals
            files = list(market_upscaled.glob('company_fundamentals_*.csv'))
            if files:
                self.datasets['company_fundamentals'] = pd.read_csv(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"  ✓ Company fundamentals: {len(self.datasets['company_fundamentals']):,} rows")
            
            # Market events
            files = list(market_upscaled.glob('market_events_*.csv'))
            if files:
                self.datasets['market_events'] = pd.read_csv(max(files, key=lambda p: p.stat().st_mtime))
                logger.info(f"  ✓ Market events: {len(self.datasets['market_events']):,} rows")
        
        # 3. FDIC BANKING (if separate from upscaled)
        banking_dir = self.data_root / 'banking'
        if banking_dir.exists():
            logger.info("\n🏦 Loading FDIC Banking...")
            
            # Active banks
            active_banks_file = banking_dir / 'active_banks.csv'
            if active_banks_file.exists():
                self.datasets['fdic_active_banks'] = pd.read_csv(active_banks_file)
                logger.info(f"  ✓ FDIC active banks: {len(self.datasets['fdic_active_banks']):,} rows")
            
            # Failed banks
            failed_banks_file = banking_dir / 'failed_banks.csv'
            if failed_banks_file.exists():
                self.datasets['fdic_failed_banks'] = pd.read_csv(failed_banks_file)
                logger.info(f"  ✓ FDIC failed banks: {len(self.datasets['fdic_failed_banks']):,} rows")
        
        # 4. OWNERSHIP
        ownership_dir = self.data_root / 'ownership'
        if ownership_dir.exists():
            logger.info("\n🏢 Loading Ownership...")
            
            # Company CIK mapping
            cik_file = ownership_dir / 'company_cik_mapping.csv'
            if cik_file.exists():
                self.datasets['company_cik'] = pd.read_csv(cik_file)
                logger.info(f"  ✓ Company CIK: {len(self.datasets['company_cik']):,} rows")
            
            # 13F holdings
            holdings_file = ownership_dir / '13f_institutional_holdings.csv'
            if holdings_file.exists():
                self.datasets['13f_holdings'] = pd.read_csv(holdings_file)
                logger.info(f"  ✓ 13F holdings: {len(self.datasets['13f_holdings']):,} rows")
            
            # Ownership network
            ownership_file = ownership_dir / 'corporate_ownership_network.csv'
            if ownership_file.exists():
                self.datasets['ownership_network'] = pd.read_csv(ownership_file)
                logger.info(f"  ✓ Ownership network: {len(self.datasets['ownership_network']):,} rows")
        
        # 5. ECONOMIC
        logger.info("\n📉 Loading Economic...")
        econ_files = list(self.data_root.glob('**/fred_economic_*.csv'))
        if econ_files:
            self.datasets['economic_indicators'] = pd.read_csv(max(econ_files, key=lambda p: p.stat().st_mtime))
            logger.info(f"  ✓ Economic indicators: {len(self.datasets['economic_indicators']):,} rows")
        
        total_rows = sum(len(df) for df in self.datasets.values())
        logger.info(f"\n✅ Loaded {len(self.datasets)} datasets with {total_rows:,} total rows")
        
        return self.datasets
    
    def create_unified_nodes(self):
        """
        Step 2: Create unified node list
        Time: 1-2 hours
        """
        logger.info("\n📊 Creating unified node list...")
        
        nodes = []
        node_id_set = set()  # Track unique node IDs
        
        # 1. BANKING INSTITUTIONS (from profiles)
        if 'institution_profiles' in self.datasets:
            df = self.datasets['institution_profiles']
            for _, inst in df.iterrows():
                node_id = f"BANK_{inst['institution_id']}"
                if node_id not in node_id_set:
                    nodes.append({
                        'node_id': node_id,
                        'node_type': 'bank',
                        'network_layer': 'banking',
                        'name': inst.get('institution_name', inst['institution_id']),
                        'institution_id': inst['institution_id'],
                        'tier': inst.get('tier', 'unknown'),
                        'total_assets': float(inst.get('total_assets', 0)),
                        'total_deposits': float(inst.get('total_deposits', 0)),
                        'total_loans': float(inst.get('total_loans', 0)),
                        'equity': float(inst.get('equity', 0)),
                        'num_branches': int(inst.get('num_branches', 0)),
                        'num_employees': int(inst.get('num_employees', 0))
                    })
                    node_id_set.add(node_id)
        
        logger.info(f"  ✓ Added {len([n for n in nodes if n['node_type'] == 'bank'])} banking nodes")
        
        # 2. FDIC BANKS (if different from upscaled)
        if 'fdic_active_banks' in self.datasets:
            df = self.datasets['fdic_active_banks']
            for _, bank in df.iterrows():
                cert = bank.get('CERT', bank.get('cert'))
                if pd.notna(cert):
                    node_id = f"FDIC_{int(cert)}"
                    if node_id not in node_id_set:
                        nodes.append({
                            'node_id': node_id,
                            'node_type': 'fdic_bank',
                            'network_layer': 'banking',
                            'name': bank.get('NAME', bank.get('name', 'Unknown')),
                            'cert': int(cert),
                            'assets': float(bank.get('ASSET', bank.get('asset', 0))) if pd.notna(bank.get('ASSET', bank.get('asset'))) else 0,
                            'city': str(bank.get('CITY', bank.get('city', ''))),
                            'state': str(bank.get('STALP', bank.get('stalp', '')))
                        })
                        node_id_set.add(node_id)
        
        # 3. MARKET STOCKS
        if 'company_fundamentals' in self.datasets:
            df = self.datasets['company_fundamentals']
            for _, company in df.iterrows():
                symbol = company.get('symbol', company.get('ticker'))
                if pd.notna(symbol):
                    node_id = f"STOCK_{symbol}"
                    if node_id not in node_id_set:
                        nodes.append({
                            'node_id': node_id,
                            'node_type': 'stock',
                            'network_layer': 'market',
                            'ticker': symbol,
                            'name': company.get('company_name', symbol),
                            'sector': company.get('sector', 'Unknown'),
                            'industry': company.get('industry', 'Unknown'),
                            'market_cap': float(company.get('market_cap', 0)),
                            'trailing_pe': float(company.get('trailing_pe', 0)) if pd.notna(company.get('trailing_pe')) else None,
                            'debt_to_equity': float(company.get('debt_to_equity', 0)) if pd.notna(company.get('debt_to_equity')) else None,
                            'beta': float(company.get('beta', 1.0)) if pd.notna(company.get('beta')) else 1.0
                        })
                        node_id_set.add(node_id)
        
        logger.info(f"  ✓ Added {len([n for n in nodes if n['node_type'] == 'stock'])} stock nodes")
        
        # 4. INSTITUTIONAL INVESTORS
        if '13f_holdings' in self.datasets:
            df = self.datasets['13f_holdings']
            unique_institutions = df['filer_cik'].unique()
            
            for cik in unique_institutions:
                if pd.notna(cik):
                    node_id = f"INST_{cik}"
                    if node_id not in node_id_set:
                        nodes.append({
                            'node_id': node_id,
                            'node_type': 'institutional_investor',
                            'network_layer': 'ownership',
                            'cik': str(cik),
                            'name': f"Institution_{cik}"
                        })
                        node_id_set.add(node_id)
        
        logger.info(f"  ✓ Added {len([n for n in nodes if n['node_type'] == 'institutional_investor'])} institutional nodes")
        
        # Convert to DataFrame
        df_nodes = pd.DataFrame(nodes)
        df_nodes['created_at'] = datetime.now().isoformat()
        
        # Save
        output_file = self.output_dir / 'unified_nodes.csv'
        df_nodes.to_csv(output_file, index=False)
        logger.info(f"\n✅ Created {len(df_nodes):,} unified nodes → {output_file}")
        
        return df_nodes
    
    def create_unified_edges(self):
        """
        Step 3: Create unified edge list
        Time: 1-2 hours
        """
        logger.info("\n🔗 Creating unified edge list...")
        
        edges = []
        edge_set = set()  # Track unique edges (source, target, type)
        
        # 1. INTERBANK LENDING EDGES
        if 'interbank_lending' in self.datasets:
            df = self.datasets['interbank_lending']
            for _, txn in df.iterrows():
                from_inst = txn.get('from_institution')
                to_inst = txn.get('to_institution')
                
                if pd.notna(from_inst) and pd.notna(to_inst):
                    source_id = f"BANK_{from_inst}"
                    target_id = f"BANK_{to_inst}"
                    rel_type = txn.get('relationship_type', 'interbank_lending')
                    
                    edge_key = (source_id, target_id, rel_type)
                    if edge_key not in edge_set:
                        edges.append({
                            'source_id': source_id,
                            'target_id': target_id,
                            'relationship_type': rel_type,
                            'network_layer': 'banking',
                            'weight': float(txn.get('amount_usd', 0)),
                            'transaction_date': txn.get('transaction_date'),
                            'maturity_days': int(txn.get('maturity_days', 0)),
                            'interest_rate': float(txn.get('interest_rate', 0)),
                            'currency': 'USD'
                        })
                        edge_set.add(edge_key)
        
        logger.info(f"  ✓ Added {len([e for e in edges if e['network_layer'] == 'banking'])} banking edges")
        
        # 2. MARKET CORRELATION EDGES
        if 'market_correlations' in self.datasets:
            df = self.datasets['market_correlations']
            for _, corr in df.iterrows():
                symbol1 = corr.get('symbol1')
                symbol2 = corr.get('symbol2')
                
                if pd.notna(symbol1) and pd.notna(symbol2):
                    source_id = f"STOCK_{symbol1}"
                    target_id = f"STOCK_{symbol2}"
                    rel_type = 'market_correlation'
                    
                    edge_key = (source_id, target_id, rel_type)
                    if edge_key not in edge_set:
                        edges.append({
                            'source_id': source_id,
                            'target_id': target_id,
                            'relationship_type': rel_type,
                            'network_layer': 'market',
                            'weight': abs(float(corr.get('correlation', 0))),
                            'correlation': float(corr.get('correlation', 0)),
                            'window_days': int(corr.get('window_days', 90))
                        })
                        edge_set.add(edge_key)
        
        logger.info(f"  ✓ Added {len([e for e in edges if e['network_layer'] == 'market'])} market edges")
        
        # 3. OWNERSHIP EDGES
        if 'ownership_network' in self.datasets:
            df = self.datasets['ownership_network']
            for _, own in df.iterrows():
                owner_cik = own.get('owner_cik')
                owned_ticker = own.get('owned_ticker')
                
                if pd.notna(owner_cik) and pd.notna(owned_ticker):
                    source_id = f"INST_{owner_cik}"
                    target_id = f"STOCK_{owned_ticker}"
                    rel_type = own.get('relationship_type', 'equity_ownership')
                    
                    edge_key = (source_id, target_id, rel_type)
                    if edge_key not in edge_set:
                        edges.append({
                            'source_id': source_id,
                            'target_id': target_id,
                            'relationship_type': rel_type,
                            'network_layer': 'ownership',
                            'weight': float(own.get('ownership_value', 0)),
                            'shares_held': int(own.get('shares_held', 0)) if pd.notna(own.get('shares_held')) else 0
                        })
                        edge_set.add(edge_key)
        
        logger.info(f"  ✓ Added {len([e for e in edges if e['network_layer'] == 'ownership'])} ownership edges")
        
        # Convert to DataFrame
        df_edges = pd.DataFrame(edges)
        df_edges['created_at'] = datetime.now().isoformat()
        
        # Save
        output_file = self.output_dir / 'unified_edges.csv'
        df_edges.to_csv(output_file, index=False)
        logger.info(f"\n✅ Created {len(df_edges):,} unified edges → {output_file}")
        
        return df_edges
    
    def create_network_statistics(self, df_nodes, df_edges):
        """
        Step 4: Generate network statistics
        Time: 30 minutes
        """
        logger.info("\n📈 Generating network statistics...")
        
        stats = {
            'timestamp': datetime.now().isoformat(),
            'total_nodes': len(df_nodes),
            'total_edges': len(df_edges),
            'layers': {},
            'datasets_used': list(self.datasets.keys())
        }
        
        # Statistics by layer
        for layer in df_nodes['network_layer'].unique():
            layer_nodes = df_nodes[df_nodes['network_layer'] == layer]
            layer_edges = df_edges[df_edges['network_layer'] == layer]
            
            stats['layers'][layer] = {
                'nodes': len(layer_nodes),
                'edges': len(layer_edges),
                'node_types': layer_nodes['node_type'].value_counts().to_dict(),
                'relationship_types': layer_edges['relationship_type'].value_counts().to_dict() if len(layer_edges) > 0 else {},
                'avg_degree': float(len(layer_edges) / len(layer_nodes)) if len(layer_nodes) > 0 else 0,
                'density': float(len(layer_edges) / (len(layer_nodes) * (len(layer_nodes) - 1))) if len(layer_nodes) > 1 else 0
            }
        
        # Save statistics
        stats_file = self.output_dir / 'network_statistics.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"✅ Saved network statistics → {stats_file}")
        
        return stats
    
    def generate_neo4j_import_files(self, df_nodes, df_edges):
        """
        Step 5: Generate Neo4j-ready import files
        Time: 30 minutes
        """
        logger.info("\n🔧 Generating Neo4j import files...")
        
        neo4j_dir = self.output_dir / 'neo4j_import'
        neo4j_dir.mkdir(exist_ok=True)
        
        # Nodes - Neo4j format
        df_nodes_neo4j = df_nodes.copy()
        df_nodes_neo4j.rename(columns={'node_id': 'node_id:ID'}, inplace=True)
        df_nodes_neo4j[':LABEL'] = df_nodes_neo4j['node_type']
        
        nodes_file = neo4j_dir / 'nodes.csv'
        df_nodes_neo4j.to_csv(nodes_file, index=False)
        logger.info(f"  ✓ Nodes → {nodes_file}")
        
        # Edges - Neo4j format
        df_edges_neo4j = df_edges.copy()
        df_edges_neo4j.rename(columns={
            'source_id': ':START_ID',
            'target_id': ':END_ID',
            'relationship_type': ':TYPE'
        }, inplace=True)
        
        edges_file = neo4j_dir / 'edges.csv'
        df_edges_neo4j.to_csv(edges_file, index=False)
        logger.info(f"  ✓ Edges → {edges_file}")
        
        logger.info(f"\n✅ Neo4j import files ready in: {neo4j_dir}")
        
        return neo4j_dir
    
    def print_summary(self, stats):
        """Print final summary"""
        print("\n" + "="*80)
        print("DATA INTEGRATION COMPLETE ✅")
        print("="*80)
        print(f"\n📊 Network Summary:")
        print(f"  Total Nodes: {stats['total_nodes']:,}")
        print(f"  Total Edges: {stats['total_edges']:,}")
        print(f"  Datasets Used: {len(stats['datasets_used'])}")
        
        print(f"\n🔍 By Layer:")
        for layer, layer_stats in stats['layers'].items():
            print(f"\n  {layer.upper()} Layer:")
            print(f"    Nodes: {layer_stats['nodes']:,}")
            print(f"    Edges: {layer_stats['edges']:,}")
            print(f"    Avg Degree: {layer_stats['avg_degree']:.2f}")
            print(f"    Density: {layer_stats['density']:.4f}")
        
        print("\n" + "="*80)
        print(f"📁 Output files in: data/integrated/")
        print("="*80)


def main():
    """
    Execute Complete Data Integration
    """
    print("="*80)
    print("COMPLETE DATA INTEGRATION FOR UPSCALED COLLECTORS")
    print("="*80)
    
    integrator = UpscaledDataIntegrator(data_root='data')
    
    # Step 1: Load all datasets
    print("\n[Step 1/5] Loading all upscaled datasets...")
    datasets = integrator.load_all_upscaled_datasets()
    
    if not datasets:
        print("❌ ERROR: No datasets found!")
        print("   Please ensure you've run all data collectors first.")
        return
    
    # Step 2: Create unified nodes
    print("\n[Step 2/5] Creating unified node list...")
    df_nodes = integrator.create_unified_nodes()
    
    # Step 3: Create unified edges
    print("\n[Step 3/5] Creating unified edge list...")
    df_edges = integrator.create_unified_edges()
    
    # Step 4: Generate statistics
    print("\n[Step 4/5] Generating network statistics...")
    stats = integrator.create_network_statistics(df_nodes, df_edges)
    
    # Step 5: Create Neo4j import files
    print("\n[Step 5/5] Generating Neo4j import files...")
    neo4j_dir = integrator.generate_neo4j_import_files(df_nodes, df_edges)
    
    # Print summary
    integrator.print_summary(stats)
    
    print("\n✅ Ready for Neo4j import!")
    print(f"   Import files: {neo4j_dir}")
    print("\n📋 Next Steps:")
    print("   1. Install Neo4j (Desktop or Docker)")
    print("   2. Run: python scripts/neo4j_import.py")
    print("   3. Verify in Neo4j Browser: http://localhost:7474")


if __name__ == "__main__":
    main()