function withdraw ( ) { if ( ! msg . sender . send ( balanceOf [ msg . sender ] ) ) revert ( ) ; }