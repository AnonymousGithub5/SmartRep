function FunctionDefinition_0 ( address _from , uint _value ) external { assert ( _from != address ( 0 ) ) ; assert ( _value > 0 ) ; assert ( Identifier_0 . transferFrom ( _from , address ( this ) , _value ) ) ; Identifier_1 [ _from ] = safeAdd ( Identifier_2 [ _from ] , _value ) ; }