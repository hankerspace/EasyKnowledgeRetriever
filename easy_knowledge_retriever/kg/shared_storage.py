from easy_knowledge_retriever.kg.state import (
    _manager,
    _is_multiprocess,
    _lock_registry,
    _lock_registry_count,
    _lock_cleanup_data,
    _registry_guard,
    _internal_lock,
    _data_init_lock,
    _shared_dicts,
    _init_flags,
    _update_flags,
    _storage_keyed_lock,
    _async_locks,
    _earliest_mp_cleanup_time,
    _last_mp_cleanup_time,
    _initialized,
    _default_workspace,
    _debug_n_locks_acquired,
)
from easy_knowledge_retriever.kg.utils import direct_log, get_final_namespace, DEBUG_LOCKS
from easy_knowledge_retriever.kg.concurrency import (
    UnifiedLock,
    KeyedUnifiedLock,
    NamespaceLock,
    _KeyedLockContext,
    get_internal_lock,
    get_storage_keyed_lock,
    get_data_init_lock,
    get_namespace_lock,
    inc_debug_n_locks_acquired,
    dec_debug_n_locks_acquired,
)
# Re-exported types for compatibility
from easy_knowledge_retriever.kg.concurrency import T
from easy_knowledge_retriever.kg.state import LockType

from easy_knowledge_retriever.kg.shared_memory import (
    initialize_share_data,
    set_default_workspace,
    get_default_workspace,
    get_namespace_data,
    try_initialize_namespace,
    initialize_pipeline_status,
    get_update_flag,
    set_all_update_flags,
    clear_all_update_flags,
)
