function FunctionDefinition_0 ( address _spender , uint Parameter_0 ) public returns ( bool ) { require ( _spender != address ( 0 ) ) ; allowed [ msg . sender ] [ _spender ] = allowed [ msg . sender ] [ _spender ] . add ( Identifier_0 ) ;