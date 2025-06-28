# アーキテクチャ設計

## 1. アーキテクチャ概要

### 1.1 設計原則
- **分散性**: 計算負荷を複数ノードに分散
- **スケーラビリティ**: ノード数の動的増減に対応
- **フォールトトレランス**: 単一ノード障害時の継続動作
- **低レイテンシ**: リアルタイム性の確保
- **一貫性**: 物理状態の整合性保証
- **イベント駆動**: イベントベースの処理

### 1.2 アーキテクチャパターン
- **マイクロサービスアーキテクチャ**: 機能別のサービス分割
- **イベントソーシング**: イベントベースのデータ管理
- **CQRS**: コマンド・クエリ責任分離
- **ストリーム処理**: リアルタイムイベント処理

## 2. システムアーキテクチャ

### 2.1 全体構成図
```
┌─────────────────────────────────────────────────────────────┐
│                    Client Layer                             │
├─────────────────────────────────────────────────────────────┤
│                 Endpoint Server                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │   Physics   │  │ Event Stream│  │   Data      │        │
│  │   Engine    │  │  Processor  │  │   Store     │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────┤
│                 Database Layer                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Event      │  │  Cache      │  │  Distributed│        │
│  │  Database   │  │  Database   │  │  Database   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘        │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 サーバー構成

#### 2.2.1 データストアサーバー
- **データ永続化**: 物理オブジェクトの状態保存
- **URI管理**: URIをキーとしたデータ管理
- **データ整合性**: データの一貫性保証
- **分散保存**: URIに記載されているホストに値が保存される
- **バックアップ**: データの冗長化

#### 2.2.2 イベントストリームサーバー
- **イベント処理**: リアルタイムイベント処理
- **ストリーム管理**: イベントストリームの管理
- **循環構造**: 数珠つなぎの循環データ構造
- **負荷分散**: イベント処理の負荷分散

#### 2.2.3 エンドポイントサーバー
- **クライアント通信**: クライアントとのTCP通信
- **API提供**: RESTful APIの提供
- **認証・認可**: セキュリティ機能
- **負荷分散**: クライアント接続の負荷分散

#### 2.2.4 クライアント
- **ユーザーインターフェース**: 操作インターフェース
- **イベント送信**: 軽量なイベント送信
- **状態表示**: 物理シミュレーションの表示
- **ローカルキャッシュ**: パフォーマンス向上

## 3. データフロー

### 3.1 イベントフロー
```
Client → Endpoint Server → Event Stream Server → Data Store Server
   ↑                                                      ↓
   └─────────────── State Update ←────────────────────────┘
```

### 3.2 データ管理フロー
```
1. ユーザー操作 → クライアント
2. クライアント → エンドポイントサーバー (TCP)
3. エンドポイントサーバー → イベントストリームサーバー
4. イベントストリームサーバー → データストアサーバー
5. データストアサーバー → 物理エンジン
6. 物理エンジン → 状態更新
7. 状態更新 → 全サーバーに配信
```

## 4. データ管理アーキテクチャ

### 4.1 URIベースデータ管理
```rust
struct DataManager {
    // URIをキーとしたデータ管理
    async fn get_data(&self, uri: &str) -> Result<PhysicsObject, Error>;
    async fn set_data(&self, uri: &str, data: &PhysicsObject) -> Result<(), Error>;
    async fn delete_data(&self, uri: &str) -> Result<(), Error>;
    async fn list_data(&self, pattern: &str) -> Result<Vec<String>, Error>;
}

// URI例
let uris = vec![
    "krapition://physics/objects/ball-001",
    "krapition://physics/objects/box-002",
    "krapition://physics/constraints/hinge-001",
    "krapition://physics/materials/metal-001"
];
```

### 4.2 分散データ保存
```rust
struct DistributedDataStore {
    // URIに基づく分散保存
    async fn save_data(&self, uri: &str, data: &PhysicsObject) -> Result<(), Error> {
        let host = self.extract_host_from_uri(uri)?;
        let database = self.get_database_for_host(&host)?;
        database.save(uri, data).await
    }
    
    async fn get_data(&self, uri: &str) -> Result<PhysicsObject, Error> {
        let host = self.extract_host_from_uri(uri)?;
        let database = self.get_database_for_host(&host)?;
        database.get(uri).await
    }
    
    fn extract_host_from_uri(&self, uri: &str) -> Result<String, Error> {
        // URIからホストを抽出
        // krapition://host:port/path の形式から host:port を取得
        let url = url::Url::parse(uri)?;
        Ok(format!("{}:{}", url.host_str().unwrap_or("localhost"), 
                   url.port().unwrap_or(8080)))
    }
}
```

### 4.3 循環データ構造
```rust
struct EventNode {
    id: String,
    timestamp: SystemTime,
    event_type: EventType,
    data: serde_json::Value,
    previous_node_id: String,
    next_node_id: String,
}

struct CircularEventStream {
    head: String,  // 最新イベントのID
    tail: String,  // 最古イベントのID
    max_size: usize,
    
    fn add_event(&mut self, event: Event) -> Result<(), Error>;
    fn get_events(&self, from_id: &str, to_id: &str) -> Result<Vec<EventNode>, Error>;
    fn cleanup(&mut self) -> Result<(), Error>;
}
```

## 5. イベントストリーム処理

### 5.1 イベントタイプ
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
enum EventType {
    // 物理イベント
    ObjectCreated,
    ObjectDeleted,
    ObjectMoved,
    CollisionDetected,
    ConstraintApplied,
    
    // 音声イベント
    SoundEmitted,
    SoundReceived,
    AcousticCollision,
    AcousticReflection,
    AcousticInterference,
    
    // ユーザーイベント
    UserInput,
    UserAction,
    UserCommand,
    
    // システムイベント
    SystemStart,
    SystemStop,
    NodeJoined,
    NodeLeft,
}
```

### 5.2 ストリーム処理エンジン
```rust
struct EventStreamProcessor {
    event_stream: CircularEventStream,
    processors: HashMap<EventType, Box<dyn EventProcessor>>,
}

impl EventStreamProcessor {
    // イベントの処理
    async fn process_event(&mut self, event: Event) -> Result<(), Error> {
        if let Some(processor) = self.processors.get(&event.event_type) {
            processor.process(&event).await?;
        }
        
        // 循環構造に追加
        self.event_stream.add_event(event)?;
        Ok(())
    }
    
    // イベントの配信
    async fn broadcast_event(&self, event: &Event) -> Result<(), Error> {
        // 全クライアントに配信
        self.broadcast_to_clients(event).await
    }
}
```

## 6. ネットワークアーキテクチャ

### 6.1 TCP通信設計
```rust
struct NetworkManager {
    // TCP接続管理
    async fn create_connection(&self, host: &str, port: u16) -> Result<Connection, Error>;
    async fn close_connection(&self, connection: &Connection) -> Result<(), Error>;
    
    // メッセージ送受信
    async fn send_message(&self, connection: &Connection, message: &Message) -> Result<(), Error>;
    fn on_message<F>(&self, callback: F) where F: Fn(Message) + Send + 'static;
}

struct Message {
    id: String,
    message_type: MessageType,
    data: serde_json::Value,
    timestamp: SystemTime,
    source: String,
    destination: String,
}
```

### 6.2 通信プロトコル
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
enum MessageType {
    // 制御メッセージ
    Connect,
    Disconnect,
    Heartbeat,
    Ack,
    Nack,
    
    // データメッセージ
    DataRequest,
    DataResponse,
    DataUpdate,
    
    // イベントメッセージ
    EventPublish,
    EventSubscribe,
    EventUnsubscribe,
}
```

## 7. データベースアーキテクチャ

### 7.1 分散データベース設計
```rust
struct DatabaseManager {
    // 分散データベース（物理データ）
    distributed_db: DistributedDatabase,
    
    // イベントデータベース（イベント履歴）
    event_db: EventDatabase,
    
    // キャッシュデータベース（高速アクセス）
    cache_db: CacheDatabase,
    
    // データの保存
    async fn save_data(&self, uri: &str, data: &serde_json::Value, db_type: DatabaseType) -> Result<(), Error>;
    
    // データの取得
    async fn get_data(&self, uri: &str, db_type: DatabaseType) -> Result<serde_json::Value, Error>;
    
    // データの削除
    async fn delete_data(&self, uri: &str, db_type: DatabaseType) -> Result<(), Error>;
}

#[derive(Debug, Clone)]
enum DatabaseType {
    Distributed,
    Event,
    Cache,
}
```

### 7.2 データベース選択戦略
- **分散DB**: 物理オブジェクトの分散保存
- **イベントDB**: イベント履歴の保存
- **キャッシュDB**: 頻繁アクセスデータの高速化

### 7.3 URIベース分散保存
```rust
struct DistributedDatabase {
    host_mapping: HashMap<String, DatabaseConnection>,
}

impl DistributedDatabase {
    async fn save(&self, uri: &str, data: &serde_json::Value) -> Result<(), Error> {
        let host = self.extract_host_from_uri(uri)?;
        let connection = self.get_connection_for_host(&host)?;
        connection.save(uri, data).await
    }
    
    async fn get(&self, uri: &str) -> Result<serde_json::Value, Error> {
        let host = self.extract_host_from_uri(uri)?;
        let connection = self.get_connection_for_host(&host)?;
        connection.get(uri).await
    }
    
    fn extract_host_from_uri(&self, uri: &str) -> Result<String, Error> {
        let url = url::Url::parse(uri)?;
        Ok(format!("{}:{}", 
                   url.host_str().unwrap_or("localhost"), 
                   url.port().unwrap_or(8080)))
    }
}
```

## 8. セキュリティアーキテクチャ

### 8.1 認証・認可
```rust
struct SecurityManager {
    // 認証
    async fn authenticate(&self, credentials: &Credentials) -> Result<AuthResult, Error>;
    
    // 認可
    async fn authorize(&self, user_id: &str, resource: &str, action: &str) -> Result<bool, Error>;
    
    // セッション管理
    async fn create_session(&self, user_id: &str) -> Result<Session, Error>;
    async fn validate_session(&self, session_id: &str) -> Result<bool, Error>;
}

struct AuthResult {
    success: bool,
    user_id: Option<String>,
    token: Option<String>,
    permissions: Option<Vec<String>>,
}
```

### 8.2 通信セキュリティ
- **TLS暗号化**: TCP通信の暗号化
- **メッセージ認証**: メッセージの整合性確認
- **レート制限**: DoS攻撃対策

## 9. スケーラビリティ設計

### 9.1 水平スケーリング
```rust
struct ScalingManager {
    // ノード追加
    async fn add_node(&self, node_config: &NodeConfig) -> Result<(), Error>;
    
    // ノード削除
    async fn remove_node(&self, node_id: &str) -> Result<(), Error>;
    
    // 負荷分散
    async fn distribute_load(&self) -> Result<(), Error>;
    
    // 健康監視
    async fn monitor_health(&self) -> Result<Vec<HealthStatus>, Error>;
}
```

### 9.2 負荷分散戦略
- **ラウンドロビン**: 順番に割り当て
- **最小接続数**: 接続数が最も少ないノード
- **重み付き**: ノードの性能に応じた重み付け
- **適応的**: 負荷に応じた動的調整

## 10. 監視・ログ設計

### 10.1 監視システム
```rust
struct MonitoringSystem {
    // メトリクス収集
    async fn collect_metrics(&self) -> Result<SystemMetrics, Error>;
    
    // アラート生成
    async fn generate_alerts(&self, metrics: &SystemMetrics) -> Result<Vec<Alert>, Error>;
    
    // ダッシュボード更新
    async fn update_dashboard(&self, metrics: &SystemMetrics) -> Result<(), Error>;
}

struct SystemMetrics {
    cpu_usage: f64,
    memory_usage: f64,
    network_latency: f64,
    event_processing_rate: f64,
    active_connections: usize,
}
```

### 10.2 ログ管理
- **構造化ログ**: JSON形式のログ
- **ログレベル**: DEBUG、INFO、WARN、ERROR
- **ログ集約**: 中央集約システム
- **ログ保持**: 適切な保持期間設定 