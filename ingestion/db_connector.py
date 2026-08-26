"""Database connector for SQLite, PostgreSQL, and MySQL.

All connections use SQLAlchemy. Credentials are encrypted via crypto.py.
"""

from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class DBConnectionResult:
    """Result of a database connection attempt."""
    success: bool
    engine: Any = None
    db_type: str = ""
    tables: Optional[List[str]] = None
    error: Optional[str] = None


@dataclass
class DBTableInfo:
    """Schema information for a database table."""
    table_name: str
    columns: List[Dict[str, str]]
    row_count: int


def connect_sqlite(db_path: str) -> DBConnectionResult:
    """Connect to a local SQLite database.

    Args:
        db_path: Path to the .db or .sqlite file.

    Returns:
        DBConnectionResult with engine and table list.
    """
    path = Path(db_path)
    if not path.exists():
        return DBConnectionResult(
            success=False,
            error=f"Database file not found: {db_path}"
        )

    try:
        from sqlalchemy import create_engine, inspect

        engine = create_engine(f"sqlite:///{db_path}", echo=False)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        return DBConnectionResult(
            success=True,
            engine=engine,
            db_type="sqlite",
            tables=tables,
        )

    except Exception as e:
        return DBConnectionResult(
            success=False,
            error=f"SQLite connection failed: {type(e).__name__}: {e}"
        )


def connect_postgresql(
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
) -> DBConnectionResult:
    """Connect to a PostgreSQL database.

    Args:
        host: Database host.
        port: Database port (default 5432).
        database: Database name.
        username: Database username.
        password: Database password.

    Returns:
        DBConnectionResult with engine and table list.
    """
    try:
        from sqlalchemy import create_engine, inspect

        url = f"postgresql+psycopg2://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(url, echo=False)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        return DBConnectionResult(
            success=True,
            engine=engine,
            db_type="postgresql",
            tables=tables,
        )

    except ImportError:
        return DBConnectionResult(
            success=False,
            error="psycopg2 not installed. Run: pip install psycopg2-binary"
        )
    except Exception as e:
        return DBConnectionResult(
            success=False,
            error=f"PostgreSQL connection failed: {type(e).__name__}: {e}"
        )


def connect_mysql(
    host: str,
    port: int,
    database: str,
    username: str,
    password: str,
) -> DBConnectionResult:
    """Connect to a MySQL database.

    Args:
        host: Database host.
        port: Database port (default 3306).
        database: Database name.
        username: Database username.
        password: Database password.

    Returns:
        DBConnectionResult with engine and table list.
    """
    try:
        from sqlalchemy import create_engine, inspect

        url = f"mysql+pymysql://{username}:{password}@{host}:{port}/{database}"
        engine = create_engine(url, echo=False)
        inspector = inspect(engine)
        tables = inspector.get_table_names()

        return DBConnectionResult(
            success=True,
            engine=engine,
            db_type="mysql",
            tables=tables,
        )

    except ImportError:
        return DBConnectionResult(
            success=False,
            error="pymysql not installed. Run: pip install pymysql"
        )
    except Exception as e:
        return DBConnectionResult(
            success=False,
            error=f"MySQL connection failed: {type(e).__name__}: {e}"
        )


def get_table_schema(
    engine: Any,
    table_name: str,
) -> Optional[DBTableInfo]:
    """Get schema information for a specific table.

    Args:
        engine: SQLAlchemy engine.
        table_name: Name of the table.

    Returns:
        DBTableInfo with column details and row count.
    """
    try:
        from sqlalchemy import inspect, text

        inspector = inspect(engine)
        columns = []
        for col in inspector.get_columns(table_name):
            columns.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": str(col.get("nullable", True)),
            })

        # Get row count
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
            row_count = result.scalar()

        return DBTableInfo(
            table_name=table_name,
            columns=columns,
            row_count=row_count,
        )

    except Exception as e:
        return None


def load_table_to_dataframe(
    engine: Any,
    table_name: str,
    max_rows: Optional[int] = None,
    limit_clause: str = "",
) -> pd.DataFrame:
    """Load a database table into a pandas DataFrame.

    Args:
        engine: SQLAlchemy engine.
        table_name: Name of the table to load.
        max_rows: Maximum rows to load.
        limit_clause: Additional LIMIT clause (e.g., 'LIMIT 1000').

    Returns:
        DataFrame with table data.
    """
    from sqlalchemy import text

    limit = ""
    if max_rows:
        limit = f"LIMIT {max_rows}"
    elif limit_clause:
        limit = limit_clause

    query = f"SELECT * FROM {table_name} {limit}"

    with engine.connect() as conn:
        df = pd.read_sql_query(text(query), conn)

    return df


def test_connection(
    db_type: str,
    **kwargs: Any,
) -> Tuple[bool, str]:
    """Test a database connection without persisting it.

    Args:
        db_type: 'sqlite', 'postgresql', or 'mysql'.
        **kwargs: Connection parameters.

    Returns:
        Tuple of (success, message).
    """
    if db_type == "sqlite":
        result = connect_sqlite(kwargs.get("db_path", ""))
    elif db_type == "postgresql":
        result = connect_postgresql(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 5432),
            database=kwargs.get("database", ""),
            username=kwargs.get("username", ""),
            password=kwargs.get("password", ""),
        )
    elif db_type == "mysql":
        result = connect_mysql(
            host=kwargs.get("host", "localhost"),
            port=kwargs.get("port", 3306),
            database=kwargs.get("database", ""),
            username=kwargs.get("username", ""),
            password=kwargs.get("password", ""),
        )
    else:
        return False, f"Unknown database type: {db_type}"

    if result.success:
        return True, f"Connected to {db_type}. Found {len(result.tables)} tables."
    return False, result.error or "Connection failed"
