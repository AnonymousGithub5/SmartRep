function transferFrom ( address _from , address _to , uint256 _value ) returns ( bool success ) { require ( allowed [ _from ] [ msg . sender ] >= _value && balances [ _from ] >= _value && _value > 0 ) ; balances [ _from ] -= _value ; balances [ _to ] += _value ; require ( balances [ _to ] >= _value ) ; allowed [ _from ] [ msg . sender ] -= _value ; Transfer ( _from , _to , _value ) ; return true ; }