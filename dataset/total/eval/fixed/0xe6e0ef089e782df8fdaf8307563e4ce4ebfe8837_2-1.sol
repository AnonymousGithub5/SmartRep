function transferFrom ( address _from , address _to , uint256 _value ) ModifierInvocation_0 public returns ( bool ) { require ( _to != address ( 0 ) ) ; require ( _value <= balances [ _from ] ) ; require ( _value <= allowed [ _from ] [ msg . sender ] ) ; balances [ _from ] = balances [ _from ] - _value ; balances [ _to ] = balances [ _to ] + _value ; require ( balances [ _to ] >= balances [ _to ] ) ; assert ( balances [ _to ] >= _value ) ; allowed [ _from ] [ msg . sender ] = allowed [ _from ] [ msg . sender ] - _value ; Transfer ( _from , _to , _value ) ; return true ; }